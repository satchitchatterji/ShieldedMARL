import sys

# from pettingzoo.mpe import simple_spread_v3

sys.path.append("../grid_envs")
import parallel_stag_hunt

sys.path.append("../pettingzoo_dilemma_envs")
from dilemma_pettingzoo import parallel_env as dilemma_parallel_env

sys.path.append("../centipede")
from centipede_env import parallel_env as centipede_env

sys.path.append("../publicgoods")
from publicgoods_env import parallel_env as public_goods_env
from publicgoodsmany_env import parallel_env as public_goodsmany_env

sys.path.append("../single_agent")
from pz_wrapper import parallel_env as pz_wrapper_parallel_env

ALL_ENVS = {
    # "simple_spread_v3": simple_spread_v3.parallel_env,
    "markov_stag_hunt": parallel_stag_hunt.parallel_env,
    "simple_stag_v0": dilemma_parallel_env,
    "simple_pd_v0": dilemma_parallel_env,
    "simple_chicken_v0": dilemma_parallel_env,
    "centipede": centipede_env,
    "publicgoods": public_goods_env,
    "publicgoodsmany": public_goodsmany_env,
    "CartSafe-v0": pz_wrapper_parallel_env,
    "GridNav-v0": pz_wrapper_parallel_env,
}

# grid_size=(5,5),stag_move_prob=0.1,rewards=(2, 10, -2),n_agents=2,n_plants=2,n_stags=1

ALL_ENVS_ARGS = {
    "simple_spread_v3": {"N": 3, "local_ratio": 0.5, "continuous_actions": False, "render_mode":None},
    "markov_stag_hunt": {"n_agents": 3, "grid_size": (10, 10), "stag_move_prob": 0.0, "rewards": (2, 10, -2), "n_plants": 2, "n_stags": 1, "render_mode":None},
    "simple_stag_v0": {"game": "stag", "num_actions": 2, "render_mode":None},
    "simple_pd_v0": {"game": "pd", "num_actions": 2, "render_mode":None},
    "simple_chicken_v0": {"game": "chicken", "num_actions": 2, "render_mode":None},
    "centipede": {"randomize_players": True, "growth_rate": 1, "render_mode":None},
    "publicgoods": {"initial_endowment": 2, "mult_factor": None, "observe_f": True, "f_params": [1, 1], "render_mode":None},
    # "publicgoods": {"initial_endowment": 2, "mult_factor": 2.5, "observe_f": False, "render_mode":None},
    "publicgoodsmany": {"n_agents":10, "initial_endowment": 10, "mult_factor": None, "observe_f": True, "f_params": [100, 0.5], "render_mode":None},
    "CartSafe-v0": {"env_name": "CartSafe-v0", "render_mode":None},
    "GridNav-v0": {"env_name": "GridNav-v0", "gridsize": 10, "render_mode":None}
}

# To add a new env:
# 1. Add the env to the ALL_ENVS dictionary with the corresponding parallel_env
# 2. Add the env to the ALL_ENVS_ARGS dictionary with the corresponding arguments
# 3. Add a shield (at least shields/{env_name}/shield_v0.pl) in the shields directory
# 4. Add a sensor wrapper in the sensor_wrappers.py file corresponding to the new shield and env