import torch
from torchvision.utils import make_grid

from captum.attr import GuidedBackprop, GuidedGradCam
import numpy as np
import matplotlib.pyplot as plt


class HookFeatures:
    def __init__(self, module):
        self.feature_hook = module.register_forward_hook(self.feature_hook_fn)

    def feature_hook_fn(self, module, input, output):
        self.features = output.clone().detach()
        self.gradient_hook = output.register_hook(self.gradient_hook_fn)

    def gradient_hook_fn(self, grad):
        self.gradients = grad

    def close(self):
        self.feature_hook.remove()
        self.gradient_hook.remove()


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, action=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.action = action

    def forward(self, obs):
        if self.action is None:
            return self.model(obs)[0]
        return self.model(obs, self.action)[0]


def compute_guided_backprop(obs, action, model):
    model = ModelWrapper(model, action=action)
    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(obs)
    return attribution

def compute_guided_gradcam(obs, action, model):
    obs.requires_grad_()
    obs.retain_grad()
    model = ModelWrapper(model, action=action)
    gbp = GuidedGradCam(model,layer=model.model.encoder.head_cnn.layers)
    attribution = gbp.attribute(obs,attribute_to_layer_input=True)
    return attribution

def compute_vanilla_grad(critic_target, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    q, q2 = critic_target(obs, action.detach())
    q.sum().backward()
    return obs.grad


def compute_attribution(model, obs, action=None,method="guided_backprop"):
    if method == "guided_backprop":
        return compute_guided_backprop(obs, action, model)
    if method == 'guided_gradcam':
        return compute_guided_gradcam(obs,action,model)
    return compute_vanilla_grad(model, obs, action)


def compute_features_attribution(critic_target, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    hook = HookFeatures(critic_target.encoder)
    q, _ = critic_target(obs, action.detach())
    q.sum().backward()
    features_gardients = hook.gradients
    hook.close()
    return obs.grad, features_gardients


def compute_attribution_mask(obs_grad, quantile=0.95):
    mask = []
    for i in [0, 3, 6]:
        #attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        attributions = obs_grad[:, i : i + 3].abs().sum(dim=1)
        q = torch.quantile(attributions.flatten(1), quantile, 1)
        mask.append((attributions >= q[:, None, None]).unsqueeze(1).repeat(1, 3, 1, 1))
    return torch.cat(mask, dim=1)


def make_obs_grid(obs, n=4):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            sample.append(obs[i, j : j + 3].unsqueeze(0))
    sample = torch.cat(sample, 0)
    return make_grid(sample, nrow=3) / 255.0


def make_attribution_pred_grid(attribution_pred, n=4):
    return make_grid(attribution_pred[:n], nrow=1)


def make_obs_grad_grid(obs_grad, n=4):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            channel_attribution, _ = torch.max(obs_grad[i, j : j + 3], dim=0)
            sample.append(channel_attribution[(None,) * 2] / channel_attribution.max())
    sample = torch.cat(sample, 0)
    q = torch.quantile(sample.flatten(1), 0.97, 1)
    sample[sample <= q[:, None, None, None]] = 0
    return make_grid(sample, nrow=3)

def compute_TID_score(obses, actor, critic, tid_path, gt_mask=None, rho_value='auto'):
    B,C,H,W = obses.shape
    S = int(C/3)
    C = int(C/S)
    tid_score = 0
    tid_var = 0
    with torch.no_grad():
        _, actions, _ , _ = actor(obses.contiguous())
    for i in range(B):
        obs = obses[i]
        action = actions[i]
        obs_np = obs.reshape(S,C,H,W).cpu().data.numpy()
        if gt_mask is not None:
            obj = gt_mask[i].reshape(S,C,H,W).cpu().data.numpy()[:,0]
        else:
            obj = 1*np.logical_and(obs_np[:,0]>obs_np[:,2], obs_np[:,1]>obs_np[:,2])

        if rho_value=='auto':
            rho = 1 - (obj.sum()/(S*H*W)).item()
            rho = np.round(rho,3)
        else:
            rho = rho_value
        obs_grad = compute_attribution(critic, obs[None], action[None].detach())
        mask = compute_attribution_mask(obs_grad, quantile=rho).float()[0]
        mask = torch.cat([mask[3*i][None] for i in range(3)],0).cpu().data.numpy()
    
        obj_overlap = obj*mask
        
        N_obj_fr = [obj[k].sum() for k in range(3)]
        N_M_fr = [int(mask[k].sum().item()) for k in range(3)]
        N_obj_M_fr = [int(obj_overlap[k].sum().item()) for k in range(3)]
            
        N_obj = sum(N_obj_fr)
        N_M = sum(N_M_fr)
        N_obj_M = sum(N_obj_M_fr)
                
        score = np.sqrt((N_obj_M * N_obj_M)/(N_obj * N_M))
            
        score_fr0 = np.sqrt((N_obj_M_fr[0] * N_obj_M_fr[0])/(N_obj_fr[0] * N_M_fr[0]))
        score_fr1 = np.sqrt((N_obj_M_fr[1] * N_obj_M_fr[1])/(N_obj_fr[1] * N_M_fr[1]))
        score_fr2 = np.sqrt((N_obj_M_fr[2] * N_obj_M_fr[2])/(N_obj_fr[2] * N_M_fr[2]))
            
        var = np.var([100*score_fr0, 100*score_fr1, 100*score_fr2])
    
        tid_score += score
        tid_var += var
            
        if i<30:
            for j in range(3):
                obs_np= obs[3*j:3*(j+1)].permute(1,2,0).cpu().data.numpy()
                obs_mask = mask[j][:,:,None]*obs_np
                plt.imsave(f'{tid_path}{i}{"_"}{"obs_"}{j}.jpg',obs_np/255)
                plt.imsave(f'{tid_path}{i}{"_"}{"obs_masked_"}{j}.jpg',obs_mask/255)
                plt.imsave(f'{tid_path}{i}{"_"}{"mask_"}{j}.jpg',np.repeat(mask[j][:,:,None], 3, axis=2))
                plt.imsave(f'{tid_path}{i}{"_"}{"gt_mask_"}{j}.jpg',np.repeat(obj[j][:,:,None], 3, axis=2))
            
    tid_score = np.round(tid_score/B,4)
    tid_var = np.round(tid_var/B,4)
    
    return tid_score, tid_var