from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.simgrl import SimGRL
from algorithms.sgsac import SGSAC
from algorithms.tlda import TLDA

algorithm = {
	'sac': SAC,
	'rad': RAD,
	'soda': SODA,
	'drq': DrQ,
	'svea': SVEA,
    'sgsac': SGSAC,
    'tlda': TLDA,
    'simgrl': SimGRL,
    'simgrl-S': SimGRL,
    'simgrl-F': SimGRL,
}


def make_agent(obs_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, action_shape, args)
