import numpy as np

class PDQAgent(object):
    def __init__(self, num_states, num_actions, num_agents):
        self.observation_type = 'discrete'
        self.action_type = 'discrete'
    
        self.q = np.zeros((num_agents, num_states, num_actions))
        self.q_visit_count = np.zeros(self.q.shape)

        self.gamma = 0.2
        self.epsilon_start = 0.6
        self.epsilon_decay = 0.9995
        self.epsilon = self.epsilon_start
        self.alpha = lambda agent, state, action: 1.0 / (1.0 + self.q_visit_count[agent, state, action]) 
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_agents = num_agents

        self.prev_states = [None]*self.num_agents
        self.prev_actions = [None]*self.num_agents
        self.rewards = [None]*self.num_agents

        self.eval_mode = False

        self.name = "Q-Learning"

        self.epsilon_histories = []
        
    def act(self, states):
        actions = []
        for agent in range(self.num_agents):
            actions.append(self.act_single(states, agent))

        self.prev_actions = actions
        self.prev_states = states

        if self.prev_actions is not None and not self.eval_mode:
            self._learn(self.prev_states, self.prev_actions, self.rewards, states, actions)
        
        return actions

    def act_single(self, states, agent):
        # shape of states: (num_agents, num_states)
        # shape of q: (num_agents, num_states, num_actions)
        if (np.random.rand() < self.epsilon or states[agent] is None) and (not self.eval_mode) :
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.q[agent, states[agent]])

        self.epsilon_histories.append(self.epsilon)
        self.epsilon *= self.epsilon_decay
        
        return action
        
    def update_reward(self, rewards):
        self.rewards = rewards

    def _learn(self, states, actions, rewards, next_states, next_actions):
        # update value function

        for agent in range(self.num_agents):
            if states[agent] is not None and rewards[agent] is not None:
                self.q_visit_count[agent, states[agent], actions[agent]] += 1
                target = rewards[agent] + self.gamma * np.max(self.q[agent, next_states[agent]])
                self.q[agent, states[agent], actions[agent]] += self.alpha(agent, states[agent], actions[agent]) * (target - self.q[agent, states[agent], actions[agent]])

    def begin_episode(self):
        self.prev_states = [None]*self.num_agents
        self.prev_actions = [None]*self.num_agents
        self.rewards = [None]*self.num_agents
        # self.epsilon = self.epsilon_start
        pass

    def get_params(self):
        # return self.q, self.epsilon, self.prev_states, self.prev_actions, self.rewards
        return self.q
    
    def set_params(self, params):   
        # self.q, self.epsilon, self.prev_states, self.prev_actions, self.rewards = params
        self.q = params
        pass