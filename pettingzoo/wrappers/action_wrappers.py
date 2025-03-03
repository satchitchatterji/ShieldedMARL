import numpy as np
import torch

def get_wrapper(env_name, num_actions, device, **kwargs):
    """
    Get the appropriate action wrapper for the given environment.
    """
    if env_name == "waterworld":
        return WaterworldActionWrapper(num_actions, device=device, **kwargs)
    else:
        return IdentityActionWrapper(num_actions, device=device, **kwargs)
    
class IdentityActionWrapper:
    """
    Default action wrapper for PettingZoo environments.
    """
    def __init__(self, num_actions, device, **kwargs):
        self.num_actions = num_actions
        self.device = device

    def __call__(self, action):
        # action = torch.tensor(action, dtype=torch.float32, device=self.device)
        return action
    
class WaterworldActionWrapper:
    """
    Wrapper for the Waterworld environment's action space.
    Waterworld has a continuous action space, but we can discretize it into finite actions for e.g. DQN.
    """
    def __init__(self, n_actions, pursuer_max_accel, max_accel_mult):
        self.pursuer_max_accel = pursuer_max_accel
        self.max_accel_mult = max_accel_mult
        self.n_actions = n_actions
        self.action_wrappers = {
            5: self.ww_action_wrapper5,
            9: self.ww_action_wrapper9,
        }

    def __call__(self, action):
        return self.action_wrappers[self.n_actions](action)

    def ww_action_wrapper9(self, action):
        move = self.pursuer_max_accel*self.max_accel_mult
        actions = np.array([
                [0, move],       # up
                [0, -move],      # down    
                [-move, 0],      # left
                [move, 0],       # right
                [move, move],    # up-right
                [move, -move],   # down-right
                [-move, move],   # up-left
                [-move, -move],  # down-left
                [0, 0],          # none
                ], dtype=np.float32)
        
        return actions[action]

    def ww_action_wrapper5(self, action):
        move = self.pursuer_max_accel*self.max_accel_mult
        actions = np.array([
                [0, move],      # up
                [0, -move],     # down
                [-move, 0],     # left
                [move, 0],      # right
                [0, 0],         # none
                ], dtype=np.float32)

        return actions[action]  