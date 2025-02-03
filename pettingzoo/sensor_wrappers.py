import torch
import numpy as np
import sys
sys.path.append("../grid_envs")
import parallel_stag_hunt as psh

from sensor_util import OnlineStats, normalized_pgg_distance

import warnings
warnings.filterwarnings("ignore")

warned = False
USE_CUDA = False

def get_wrapper(env):
    wrappers = {
        "waterworld": WaterworldSensorWrapper,
        "simple_stag_v0": StagHuntSensorWrapper,
        "simple_pd_v0": IdentitySensorWrapper,
        "simple_chicken_v0": IdentitySensorWrapper,
        "markov_stag_hunt": MarkovStagHuntSensorWrapper,
        "centipede": IdentitySensorWrapper,
        "publicgoods": PublicGoodsSensorWrapper,
        "publicgoodsmany": PublicGoodsManySensorWrapper,
        "CartSafe-v0": CartSafeSensorWrapper,
        "GridNav-v0": GridNavSensorWrapper,
    }
    if env in wrappers.keys():
        return wrappers[env]
    else:
        raise NotImplementedError(f"Sensor wrapper for {env} not implemented.")

class Wrapper:
    """
    Abstract wrapper class
    """
    def __init__(self, env, num_sensors=None, device=None):
        self.env = env
        self.device = device
        if num_sensors is not None:
            self.num_sensors = num_sensors
        else:
            self.num_sensors = env.observation_spaces[env.agents[0]].shape[0]

        self.translation_func = None
    
    def __call__(self, x):
        raise NotImplementedError("Wrapper not implemented.")

class IdentitySensorWrapper(Wrapper):
    """
    Sensor wrapper for the PettingZoo environments. 
    """
    def __init__(self, env, num_sensors=None, device=None):
        super().__init__(env, num_sensors=num_sensors, device=device)
    
    def __call__(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self.device)
    
class StagHuntSensorWrapper(Wrapper):
    """
    Sensor wrapper for the Stag Hunt environment. 
    """
    def __init__(self, env, num_sensors=None, device=None):
        super().__init__(env, num_sensors=num_sensors, device=device)
        self.translation_func = self.stag_counter

        self.buffer_size = 50
        self.stag_counter = [list(), list()]
        self.hare_counter = [list(), list()]
        self.ticker = 0

    def stag_counter(self, obs):
        global warned

        obs = obs.cpu().numpy()
        agent_identity = int((self.ticker*2)%2)
        self.ticker += 0.5 # increment by 0.5 since there are two agents in the environment
        self.stag_counter[agent_identity].append(obs[0])
        self.hare_counter[agent_identity].append(obs[1])

        if len(self.stag_counter[agent_identity]) > self.buffer_size:
            self.stag_counter[agent_identity].pop(0)
            self.hare_counter[agent_identity].pop(0)
        
        true_eq = [0.6,0.4]
        if not warned:
            print(">>> WARNING: (sensor_wrappers.py) Using hardcoded true equilibrium for Stag Hunt environment: ", true_eq, " <<<")
            warned = True

        stag_percent = np.mean(self.stag_counter[agent_identity])
        hare_percent = np.mean(self.hare_counter[agent_identity])

        # print(f"Agent: {agent_identity}, Stag: {stag_percent}, Hare: {hare_percent}")

        stag_diff = np.abs(stag_percent - true_eq[0])
        hare_diff = np.abs(hare_percent - true_eq[1])

        return torch.tensor(np.array([stag_diff, hare_diff], dtype=np.float32), dtype=torch.float32, device=self.device)
        
    def __call__(self, x):
        # TODO: batch processing
        if len(x.shape) == 1:
            return self.translation_func(x)
        else:
            return torch.stack([self.translation_func(x[i]) for i in range(x.shape[0])])
        
class PublicGoodsSensorWrapper(Wrapper):
    """
    Sensor wrapper for the Public Goods Game environment. 
    """
    def __init__(self, env, num_sensors=None, device=None):
        super().__init__(env, num_sensors=num_sensors, device=device)

        if env.observe_f:
            self.translation_func = self.obs_with_f
        else:
            self.translation_func = self.obs_without_f
            
        self._reset_values()

    def _reset_values(self):
        self.stats = OnlineStats()
        self.ticker = 0

    def obs_without_f(self, obs):
        return torch.tensor(obs, dtype=torch.float32, device=self.device)
    
    def obs_with_f(self, obs):
        if self.env.num_moves == 0:
            self._reset_values()

        obs = obs.cpu().numpy()
        agent_other = obs[:-1]
        agent_identity = int((self.ticker*2)%2)

        mult = obs[-1] # todo: update this to pdf
        if agent_identity == 0 and self.ticker > 0:
            self.stats.update(mult)
        self.ticker += 0.5
        if self.ticker > 2:
            mult_uncertainty = normalized_pgg_distance(mult, self.stats.mean, self.stats.stddev())
        else:
            mult_uncertainty = 1
        mult_high = self.stats.mean > 1
        returnval = np.hstack([agent_other, [mult_uncertainty, mult_high]]).astype(dtype=np.float32)
        # print(self.stats.mean, self.stats.stddev(), returnval)
        return torch.tensor(returnval, dtype=torch.float32, device=self.device)
    
    def __call__(self, x):
        # TODO: batch processing
        if len(x.shape) == 1:
            return self.translation_func(x)
        else:
            return torch.stack([self.translation_func(x[i]) for i in range(x.shape[0])])

class PublicGoodsManySensorWrapper(Wrapper):
    """
    Sensor wrapper for the Public Goods Game environment. 
    """
    def __init__(self, env, num_sensors=None, device=None):
        super().__init__(env, num_sensors=num_sensors, device=device)
        
        self.num_sensors = 1
        if env.observe_f:
            self.translation_func = self.obs_with_f
        else:
            self.translation_func = self.obs_without_f

    def obs_without_f(self, obs):
        return torch.tensor(obs, dtype=torch.float32, device=self.device)
    
    def obs_with_f(self, obs):
        obs = obs.cpu().numpy()
        agent_other = obs[0]
        # return % of cooperating agents
        return torch.tensor(agent_other, dtype=torch.float32, device=self.device)

    def __call__(self, x):
        # TODO: batch processing
        if len(x.shape) == 1:
            return self.translation_func(x)
        else:
            return torch.stack([self.translation_func(x[i]) for i in range(x.shape[0])])

class CartSafeSensorWrapper(Wrapper):
    """
    Sensor wrapper for the CartSafe environment. 
    """
    def __init__(self, env, num_sensors=None):
        raise NotImplementedError("CartSafeSensorWrapper not implemented.")
        self.env = env
        self.num_sensors = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
        self.translation_func = self.cart_safe_simple
    
    def cart_safe_simple(self, obs):
        """
        Returns normalized values of the agent's position and cost
        """
        obs = obs.cpu().numpy()
        x = obs[0]

        x_cost = np.abs(x) > self.env._env.x_constraint
        x_pos = np.abs(x-self.env._env.x_threshold/2)  / (self.env._env.x_threshold/2)
        left = x < 0

        return torch.tensor(np.array([x_cost, x_pos, left, 1-left], dtype=np.float32), dtype=torch.float32, device=self.device)

    def __call__(self, x):
        # TODO: batch processing
        if len(x.shape) == 1:
            return self.translation_func(x)
        else:
            return torch.stack([self.translation_func(x[i]) for i in range(x.shape[0])])


class GridNavSensorWrapper:
    """
    Sensor wrapper for the GridNav environment. 
    """
    def __init__(self, env, num_sensors=None):
        raise NotImplementedError("GridNavSensorWrapper not implemented.")
        self.env = env
        self.num_sensors = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
        self.translation_func = self.grid_nav_simple
        self.obstruction_map = self.env._env.obstacle_states
    
    def grid_nav_simple(self, obs):
        """
        Returns if an obstacle is in the agent's path
        """
        obs = obs.cpu().numpy()

        agent_cur_pos = np.argmax(obs)
        agent_x, agent_y = agent_cur_pos // self.env._env.gridsize, agent_cur_pos % self.env._env.gridsize
        
        # check if there is an obstruction in the cardinal directions compared to self.obstruction_map

        down = list(self.env._env.ACTIONS[0] + np.array([agent_x, agent_y])) in self.obstruction_map
        left = list(self.env._env.ACTIONS[1] + np.array([agent_x, agent_y])) in self.obstruction_map
        up = list(self.env._env.ACTIONS[2] + np.array([agent_x, agent_y])) in self.obstruction_map
        right = list(self.env._env.ACTIONS[3] + np.array([agent_x, agent_y])) in self.obstruction_map

        return torch.tensor(np.array([down, left, up, right], dtype=np.float32), dtype=torch.float32, device=self.device)

    def __call__(self, x):
        # TODO: batch processing
        if len(x.shape) == 1:
            return self.translation_func(x)
        else:
            return torch.stack([self.translation_func(x[i]) for i in range(x.shape[0])])



class MarkovStagHuntSensorWrapper(Wrapper):

    def __init__(self, env, num_sensors=None, device=None):
        super().__init__(env, num_sensors=num_sensors, device=device)

        self.num_sensors = 6 # based on stag relative position
        self.translation_func = self.stag_surrounded
        self.env = env

    def one_hot_to_obs(self, one_hot_obs):
        GRID_SIZE = self.env.grid_size
        one_hot_obs = torch.tensor(one_hot_obs, dtype=torch.float32, device=self.device)
        one_hot_obs = one_hot_obs.reshape((GRID_SIZE[0], GRID_SIZE[1], psh.N_OBS_TYPES))
        obs = torch.zeros((GRID_SIZE[0], GRID_SIZE[1]), dtype=torch.int32, device=self.device)
        for x in range(GRID_SIZE[0]):
            for y in range(GRID_SIZE[1]):
                obs[x][y] = torch.argmax(one_hot_obs[x][y]).item()
        return obs
    
    def stag_surrounded(self, obs):
        """
        Returns the relative position of the stag to the agent,
                    if the agent is close to the stag, and
                    if the stag is surrounded by both agents.
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = self.one_hot_to_obs(obs)
        stag_x, stag_y = torch.where(obs == psh.OBS_STAG)
        stag_x, stag_y = stag_x[0], stag_y[0]

        agent_self = torch.where(obs == psh.OBS_AGENT_SELF)
        agent_other = torch.where(obs == psh.OBS_AGENT_OTHER)
        both_agents = torch.where(obs == psh.OBS_AGENT_BOTH)
        try:
            agent_self = agent_self if len(agent_self[0]) > 0 else both_agents
            agent_self_x, agent_self_y = agent_self[0][0], agent_self[1][0]

            agent_other = agent_other if len(agent_other[0]) > 0 else both_agents
            agent_other_x, agent_other_y = agent_other[0][0], agent_other[1][0]
        except IndexError as e:
            print(obs)
            print(agent_self, agent_other, both_agents)
            raise e
        # print("\n", stag_x, stag_y, agent_self_x, agent_self_y, agent_other_x, agent_other_y, both_agents, "\n")
        assert agent_self_x is not None, f"Agent self not found in observation: {obs}"

        rel_x = stag_x - agent_self_x
        rel_y = stag_y - agent_self_y
        
        stag_above = rel_x < 0
        stag_below = rel_x > 0
        stag_left = rel_y < 0
        stag_right = rel_y > 0

        # check which agents are in the stag's Moore's neighborhood
        stag_near_self = 0
        if agent_self_x-1 <= stag_x <= agent_self_x+1 and agent_self_y-1 <= stag_y <= agent_self_y+1:
            stag_near_self = 1

        stag_near_other = 0
        if agent_other_x-1 <= stag_x <= agent_other_x+1 and agent_other_y-1 <= stag_y <= agent_other_y+1:
            stag_near_other = 1

        return torch.tensor([stag_left, stag_right, stag_above, stag_below, stag_near_self, stag_near_other], device=self.device, dtype=torch.int32)
        # return torch.tensor(np.array([stag_left, stag_right, stag_above, stag_below, stag_near_self, stag_near_other], dtype=np.int32), 
        #                     dtype=torch.int32, device=self.device)

    def __call__(self, x):
        # TODO: batch processing
        if len(x.shape) == 1:
            return self.translation_func(x)
        else:
            return torch.stack([self.translation_func(x[i]) for i in range(x.shape[0])])
        
class WaterworldSensorWrapper:
    """
    Sensor wrapper for the Waterworld environment. 
    This is to convert the sensor values to a (probabalistic) form that the shield can use.
    """
    def __init__(self, env, output_type="invert"):
        raise NotImplementedError("WaterworldSensorWrapper not implemented.")
        self.env = env
        self.num_inputs = env.observation_spaces[env.agents[0]].shape[0]
        self.output_type = output_type
        self.device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
        if output_type == "invert":
            self.num_sensors = self.num_inputs
            self.translation_func = self.invert_sensor_vals
        elif output_type == "reduce_to_8":
            self.num_sensors = 8*5+2
            self.translation_func = self.reduce_to_8
        else:
            raise NotImplementedError("Only 'invert' and 'reduce_to_8' is supported for now.")            
    
    def reduce_to_8(self, x, use_cuda=True):
        assert len(x) == 162, f"Expected 162 sensor values, got {len(x)}"
        x_new = torch.zeros(8*5+2, device=self.device) # 5 types of sensors, 8 ranges, 2 for the last two sensors
        x_new[-1] = 1-x[-1]
        x_new[-2] = 1-x[-2]
        x_diff = 1-x[:len(x)-2]
        x_diff = x_diff.reshape((32, -1)) # 32 x 5
        ranges = [[-2,2], [2,6], [6,10], [10,14], [14,18], [18,22], [22,26], [26,30], [30,31]] # 8 ranges
        ranged_x_diff = []
        first_range = torch.stack(list(x_diff[-2:0])+list(x_diff[0:3])+list(x_diff[30:]), axis=-1) 
        ranged_x_diff.append(first_range)
        for r in ranges[1:-1]:
            ranged_x_diff.append(x_diff[r[0]:r[1]+1])
        ranged_x_diff = torch.stack(ranged_x_diff).to(self.device)
        ranged_x_diff = ranged_x_diff.mean(axis=1)
        x_new[:8*5] = ranged_x_diff.flatten()
        return x_new

    def invert_sensor_vals(self, x):
        x[:len(x)-2] = 1-x[:len(x)-2]
        return x
    
    def __call__(self, x):
        # TODO: batch processing
        if len(x.shape) == 1:
            return self.translation_func(x)
        else:
            return torch.stack([self.translation_func(x[i]) for i in range(x.shape[0])])