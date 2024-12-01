import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.datasets as datasets
import kornia
import utils
import os
import torchvision

import matplotlib.pyplot as plt


places_dataloader = None
places_iter = None


def _load_places(batch_size=256, image_size=84, num_workers=16, use_val=False):
    global places_dataloader, places_iter
    partition = 'val' if use_val else 'train'
    print(f'Loading {partition} partition of places365_standard...')
    for data_dir in utils.load_config('datasets'):
        if os.path.exists(data_dir):
            fp = os.path.join(data_dir, 'places365_standard', partition)
            if not os.path.exists(fp):
                print(f'Warning: path {fp} does not exist, falling back to {data_dir}')
                fp = data_dir
            places_dataloader = torch.utils.data.DataLoader(
				datasets.ImageFolder(fp, TF.Compose([
					TF.RandomResizedCrop(image_size),
					TF.RandomHorizontalFlip(),
					TF.ToTensor()
				])),
				batch_size=batch_size, shuffle=False,
				num_workers=num_workers, pin_memory=True)
            places_iter = iter(places_dataloader)
            break
    if places_iter is None:
        raise FileNotFoundError('failed to find places365 data at any of the specified paths')
    print('Loaded dataset from', data_dir)


def _get_places_batch(batch_size):
    global places_iter
    try:
        imgs, _ = next(places_iter)
        if imgs.size(0) < batch_size:
            places_iter = iter(places_dataloader)
            imgs, _ = next(places_iter)
    except StopIteration:
        places_iter = iter(places_dataloader)
        imgs, _ = next(places_iter)
    return imgs.cuda()


def random_overlay(x, alpha=0.5, dataset='places365_standard'):
    """Randomly overlay an image from Places"""
    global places_iter
    
    if dataset == 'places365_standard':
        if places_dataloader is None:
            _load_places(batch_size=x.size(0), image_size=x.size(-1))
        imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1)//3, 1, 1)
    else:
        raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')
        
    return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.

def random_overlay_shift(x, alpha=0.5, SRO_l = 6, dataset='places365_standard'):
    """Randomly overlay an image from Places"""
    global places_iter
    n_stack = x.size(1)//3
    
    if dataset == 'places365_standard':
        if places_dataloader is None:
            img_size = x.size(-1)+2*n_stack*SRO_l
            _load_places(batch_size=x.size(0), image_size=img_size)
        imgs_orig = _get_places_batch(batch_size=x.size(0)).repeat(n_stack, 1, 1, 1)
        imgs_shifted = random_crop_shift(imgs_orig, size=84, n_stack=n_stack, shift_max=SRO_l)
        
        imgs = imgs_shifted.clone()
        imgs = imgs.permute(1,0,2,3,4).reshape(x.size(0), x.size(1), x.size(2), x.size(3))
        
    else:
        raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')
        
    return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.

def attribution_augmentation(x, mask, dataset="places365_standard"):
    """Complete non importnant pixels with a random image from Places"""
    global places_iter

    if dataset == "places365_standard":
        if places_dataloader is None:
            _load_places(batch_size=x.size(0), image_size=x.size(-1))
        imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1) // 3, 1, 1)
    else:
        raise NotImplementedError(
            f'overlay has not been implemented for dataset "{dataset}"'
        )

    # s_plus = random_conv(x) * mask
    s_plus = x * mask
    s_tilde = (((s_plus) / 255.0) + (imgs * (torch.ones_like(mask) - mask))) * 255.0
    s_minus = imgs * 255
    return s_tilde

def random_conv(x):
	"""Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
	n, c, h, w = x.shape
	for i in range(n):
		weights = torch.randn(3, 3, 3, 3).to(x.device)
		temp_x = x[i:i+1].reshape(-1, 3, h, w)/255.
		temp_x = F.pad(temp_x, pad=[1]*4, mode='replicate')
		out = torch.sigmoid(F.conv2d(temp_x, weights))*255.
		total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
	return total_out.reshape(n, c, h, w)


def batch_from_obs(obs, batch_size=32):
	"""Copy a single observation along the batch dimension"""
	if isinstance(obs, torch.Tensor):
		if len(obs.shape)==3:
			obs = obs.unsqueeze(0)
		return obs.repeat(batch_size, 1, 1, 1)

	if len(obs.shape)==3:
		obs = np.expand_dims(obs, axis=0)
	return np.repeat(obs, repeats=batch_size, axis=0)


def prepare_pad_batch(obs, next_obs, action, batch_size=32):
	"""Prepare batch for self-supervised policy adaptation at test-time"""
	batch_obs = batch_from_obs(torch.from_numpy(obs).cuda(), batch_size)
	batch_next_obs = batch_from_obs(torch.from_numpy(next_obs).cuda(), batch_size)
	batch_action = torch.from_numpy(action).cuda().unsqueeze(0).repeat(batch_size, 1)

	return random_crop_cuda(batch_obs), random_crop_cuda(batch_next_obs), batch_action


def identity(x):
	return x


def random_shift(imgs, pad=4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _,_,h,w = imgs.shape
    imgs = F.pad(imgs, (pad, pad, pad, pad), mode='replicate')
	#return kornia.augmentation.RandomCrop((h, w))(imgs)
    return random_crop(imgs, size=84)

def random_crop(x, size=84, w1=None, h1=None, return_w1_h1=False):
	"""Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
	assert (w1 is None and h1 is None) or (w1 is not None and h1 is not None), \
		'must either specify both w1 and h1 or neither of them'
	assert isinstance(x, torch.Tensor) and x.is_cuda, \
		'input must be CUDA tensor'
	
	n = x.shape[0]
	img_size = x.shape[-1]
	crop_max = img_size - size

	if crop_max <= 0:
		if return_w1_h1:
			return x, None, None
		return x

	x = x.permute(0, 2, 3, 1)

	if w1 is None:
		w1 = torch.LongTensor(n).random_(0, crop_max)
		h1 = torch.LongTensor(n).random_(0, crop_max)

	windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0,:,:, 0]
	cropped = windows[torch.arange(n), w1, h1]

	if return_w1_h1:
		return cropped, w1, h1

	return cropped

def random_crop_shift(x, size=84, n_stack=3, shift_max=6):
    """Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
        
    n = x.shape[0]
    c = x.shape[1]
    b = n//n_stack
    img_size = x.shape[-1]
    pad = (img_size - size)//2
        
    x = x.permute(0, 2, 3, 1)
    
    h1 = torch.zeros(n).type(torch.LongTensor)
    w1 = torch.zeros(n).type(torch.LongTensor)
    h0 = torch.ones(b).type(torch.LongTensor)*pad
    w0 = torch.ones(b).type(torch.LongTensor)*pad
    dw = torch.LongTensor(b).random_(-shift_max, shift_max+1)
    dh = torch.LongTensor(b).random_(-shift_max, shift_max+1)
    
    for i in range(n_stack):
        h1[b*i:b*(i+1)] = h0 + dh*i
        w1[b*i:b*(i+1)] = w0 + dw*i
    
    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0,:,:, 0]
    cropped = windows[torch.arange(n), w1, h1]
        
    return cropped.reshape(n_stack,b,c,size,size)

def random_cutout(imgs, min_cut=20,max_cut=40):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        min / max cut: int, min / max size of cutout 
        returns np.array
    """

    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = torch.empty((n, c, h, w), dtype=imgs.dtype).to(imgs.device)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.clone()
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
        #print(img[:, h11:h11 + h11, w11:w11 + w11].shape)
        cutouts[i] = cut_img
    return cutouts

def choose_avg(obs, aim_attention, out):

	avg_aim_attention = torchvision.transforms.Resize(84)(aim_attention).cuda()
	mean_region = torch.mean(avg_aim_attention.view(avg_aim_attention.shape[0], -1), dim=1)
	threshold = mean_region
	threshold_matrix = threshold.reshape(avg_aim_attention.shape[0], 1, 1).repeat(1, 84, 84)
	chose_region = torch.where(avg_aim_attention > threshold_matrix, 1., 0.)
	final_out = chose_region
	final_out = final_out.unsqueeze(1).repeat(1, 9, 1, 1)

	final_out = final_out * obs + (1 - final_out) * out

	return final_out


def view_as_windows_cuda(x, window_shape):
	"""PyTorch CUDA-enabled implementation of view_as_windows"""
	assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
		'window_shape must be a tuple with same number of dimensions as x'
	
	slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
	win_indices_shape = [
		x.size(0),
		x.size(1)-int(window_shape[1]),
		x.size(2)-int(window_shape[2]),
		x.size(3)    
	]

	new_shape = tuple(list(win_indices_shape) + list(window_shape))
	strides = tuple(list(x[slices].stride()) + list(x.stride()))

	return x.as_strided(new_shape, strides)
