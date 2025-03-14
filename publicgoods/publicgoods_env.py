# amended from https://github.com/arjun-prakash/pz_dilemma/blob/main/src/environments/centipede/centipede_v1.py

from gym import spaces
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers

import functools

MOVES = {'DEFECT': 0, 'COOPERATE': 1, 'NONE': 2}

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = parallel_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class parallel_env(ParallelEnv):
    """ 2-player public goods game """

    metadata = {'render.modes': ['human'], "name": "publicgoods", "is_parallelizable": True}

    def __init__(self, n_agents=2, initial_endowment=1, max_cycles=50,  mult_factor=None, observe_f=True, f_params=None, render_mode=None):
        self.agents = ["player_" + str(r) for r in range(n_agents)]
        assert n_agents > 1, "Number of agents must be greater than 1."
        if n_agents > 2:
            raise NotImplementedError("Only 2 agents are currently supported.")
        self.possible_agents = self.agents[:]
        self.max_cycles = max_cycles
        self.observe_f = observe_f

        self.static_mult_factor = mult_factor
        self.f_params = f_params
        self.rand_mult_factor = None
        self._mult_factor = None

        # print('Public Goods Game!')
        if self.static_mult_factor is not None and self.f_params is not None:
            print("[EPGG] Warning: both mult_factor and f_params are set. Using mult_factor.")
            self.f_params = None
            self._mult_factor = self.static_mult_factor
        elif self.static_mult_factor is None and self.f_params is None:
            print("[EPGG] Error: neither mult_factor nor f_params are set! Exiting...")
            exit(1)

        self.state = {agent: MOVES["NONE"] for agent in self.agents}
        self.initial_endowment = initial_endowment  # Initial value of endowment
        self.render_mode = render_mode

        self.action_spaces = {agent: spaces.Discrete(len(MOVES) - 1) for agent in self.agents}
        self.observation_spaces = {agent: spaces.Discrete(len(MOVES)) for agent in self.agents}

        self.reset()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if self.observe_f:
            return spaces.Discrete(len(MOVES)+1)
        return spaces.Discrete(len(MOVES))
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(len(MOVES)-1)
    
    def gen_mult_factor(self):
        if self.static_mult_factor is not None:
            self._mult_factor = self.static_mult_factor
        else:
            self._mult_factor = np.random.normal(*self.f_params)
        
    def reset(self):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.gen_mult_factor()
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.observations = {agent: 2 for agent in self.agents}
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        self.num_moves = 0
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: MOVES["NONE"] for agent in self.agents}
        
        return self.observations, self.infos

    def render(self):
        if self.render_mode == "human":
            print(f"Player actions: {self.state}")
            
    def observe(self, agent):
        flat_obs = np.zeros(len(MOVES))
        flat_obs[self.observations[agent]] = 1
        if self.observe_f:
            flat_obs = np.append(flat_obs, self._mult_factor)
        return flat_obs

    def close(self):
        pass

    def step(self, actions):
        self.rewards = {agent: 0 for agent in self.agents}
        env_truncated = False
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        self.num_moves += 1

        total_pot = np.sum(action == MOVES['COOPERATE'] for action in actions.values()) * self._mult_factor * self.initial_endowment
        divided_pot = total_pot / len(self.agents)

        for agent in self.agents:
            action = actions[agent]
            self.state[agent] = action

            if action == MOVES['COOPERATE']:
                self.rewards[agent] = divided_pot

            elif action == MOVES['DEFECT']:
                self.rewards[agent] = divided_pot + self.initial_endowment

        if self.num_moves >= self.max_cycles:
            self.terminations = {agent: True for agent in self.agents}
            env_truncated = True

        self.gen_mult_factor()
        self.observations = {agent: self.state[self.agents[1 - self.agents.index(agent)]] for agent in self.agents}
        self.observations = {agent: self.observe(agent) for agent in self.agents}

        self._accumulate_rewards()
        
        if env_truncated:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def _accumulate_rewards(self):
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]