from .base import BaseMARLAlgo
from .ppo_shielded import PPOShielded
import numpy as np

class SIPPO(BaseMARLAlgo):
    def __init__(self, env, observation_space, n_discrete_actions, action_wrapper, algorithm_params, sh_params, alpha, shielded_ratio=1.0, **kwargs):
        super().__init__(env=env, 
                         observation_space=observation_space, 
                         n_discrete_actions=n_discrete_actions, 
                         action_wrapper=action_wrapper,
                         algorithm_params=algorithm_params,
                         sh_params=sh_params,
                         alpha=alpha,
                         shielded_ratio=shielded_ratio,
                         **kwargs
                         )
        self.sensor_wrapper = lambda x: x
        
    def create_agents(self):
        for agent in self.env.agents:
            self.agents[agent] = PPOShielded(state_dim=self.observation_space, 
                                            action_dim=self.n_discrete_actions, 
                                            alpha=self.alpha,
                                            policy_safety_params=self.sh_params,
                                            policy_kw_args={"shield_params":self.sh_params, "get_sensor_value_ground_truth":self.sensor_wrapper},
                                            **self.algorithm_params)
        
        if self.shielded_ratio == 1.0:
            return
        
        n_unshielded = np.round((1-self.shielded_ratio) * len(self.env.agents))
        n_shielded = len(self.env.agents) - n_unshielded
        print(f"[SIPPO INFO]: Creating {n_shielded} shielded ({n_unshielded} unshielded) agents out of {len(self.env.agents)} total agents.")
        print(f"              Shielded ratio (requested): {self.shielded_ratio} | Shielded ratio (actual): {n_shielded/len(self.env.agents)}")

        unshielded_agents = np.random.choice(self.env.agents, int(n_unshielded), replace=False)
        for agent in unshielded_agents:
            self.agents[agent] = PPOShielded(state_dim=self.observation_space, 
                                             action_dim=self.n_discrete_actions, 
                                             policy_kw_args={"get_sensor_value_ground_truth":self.sensor_wrapper},
                                            **self.algorithm_params)
