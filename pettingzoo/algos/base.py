from abc import abstractmethod

class BaseMARLAlgo:
    def __init__(self, 
                 env, 
                 observation_space, 
                 n_discrete_actions, 
                 action_wrapper=None, 
                 sensor_wrapper=None, 
                 sh_params=None, 
                 algorithm_params=None,
                 alpha=None
                 ):
        self.agents = {}
        self.env = env
        self.observation_space = observation_space
        self.n_discrete_actions = n_discrete_actions
        self.action_wrapper = action_wrapper
        self.sensor_wrapper = sensor_wrapper
        self.sh_params = sh_params
        self.algorithm_params = algorithm_params
        self.alpha = alpha
        self.create_agents()

    @abstractmethod
    def create_agents(self):    
        raise NotImplementedError
    
    def act(self, observations):
        actions = {}
        for agent in self.env.agents:
            actions[agent] = self.action_wrapper(self.agents[agent].act(observations[agent]))

        return actions
    
    def update_rewards(self, rewards, terminations, truncations):
        for agent in self.agents.keys():
            self.agents[agent].update_reward(rewards[agent], terminations[agent] or truncations[agent])

    def eval(self, bool_val):
        for agent in self.agents.keys():
            self.agents[agent].set_eval_mode(bool_val)