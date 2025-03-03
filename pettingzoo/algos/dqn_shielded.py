import numpy as np
import random

# from playground.marl_nfg.pd_dqn_agent import PDDQNAgent
from pls.shields.shields import Shield
from torch.distributions import Categorical
import torch

from .utils import compute_eps_min_timestep

# Used as keys for the agent's memory
STATE = 0
Q_VALS = 1
ACTION = 2
REWARD = 3
TERMINAL = 4
ACTION_DIST = 5
# SAFETY_DIST = 6

class DQNShielded(object):
    """ Agent that uses the SARSA update rule to learn Q(s,a) estimates,
        using an MLP as a function approximator. """
    def __init__(self, 
                 num_states, 
                 num_actions,
                 func_approx=None, 
                 shield_params=None, 
                 shield=None, 
                 alpha=1.0,
                 train_epochs=50,
                 gamma=0.999,
                 lr=0.01,
                 batch_size=128,
                 buffer_size=10000,
                 eps_min=0.1,
                 eps_decay=0.999,
                 update_timestep=25,
                 tau=0.01,
                 update_target_type='soft',
                 explore_policy='e_greedy',
                 eval_policy='greedy',
                 on_policy=False,
                 device='none',
                 **kwargs # made to be compatible with PPO
                 ):
        
        self.device = device
        print(f"[DQN INFO] Using device: {self.device}")

        self.observation_type = 'discrete'
        self.action_type = 'discrete'
        self.learning = True

        self.step = 0
        self.on_policy = on_policy

        # input: num_states
        # output: Q(s,a)
        self.num_states = num_states
        self.num_actions = num_actions

        # set up agent
        self.n_inputs = None
        self.controls = list(range(self.num_actions))
        self.training = True
        self.eval_mode = False
    
        self.func_approx = None
        self.optimizer = None
        self.base_loss_fn = None
        if func_approx is not None:
            self.func_approx = func_approx.to(self.device)
            
        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_decay = eps_decay
        self.epsilon_min = eps_min
        self.explore_policy = explore_policy
        self.eval_policy = eval_policy

        self.gamma = gamma
        self.learning_rate = lr

        self.max_history = buffer_size
        self.batch_size = batch_size
        self.epochs = train_epochs
        self.update_timestep = update_timestep
        self.update_target_type = update_target_type
        self.tau = tau

        # memory and bookkeeping
        self.history = []
        self._temp_history = []
        self.saved_model_name = ""
        self.name = "DQNShielded"

        # initialize memory for now 
        # TODO: merge functionality with self.history later
        self.prev_state = None
        self.prev_action = None
        self.reward = None

        # shield
        # agents can share a shield if a shield is passed in
        assert shield_params is None or shield is None, "Cannot pass in both shield_params and shield"
        self.shield = None
        if shield_params is not None:
            self.shield = Shield(**shield_params)
        elif shield is not None:
            self.shield = shield
        elif shield_params is None and shield is None:
            self.shield = None

        # TODO: add shield to gpu
        # if self.shield is not None:
        #     self.shield.shield_layer.to(self.device)

        self.alpha = alpha

        self.debug_info_history = []
        self.loss_info = []
        self.save_debug_info = False
        if self.explore_policy == 'e_greedy':
            print(f"[DQN INFO] Epsilon min {eps_min} will be reached at timestep:", compute_eps_min_timestep(1.0, eps_min, eps_decay))

        self._setup()

    def _setup(self):
        """ Set up the agent for training or evaluation """

        # bookkeeping
        self.prev_state = None
        self.prev_action = None
        self.reward = None
        # set up function approximator
        self.n_inputs = self.num_states
        self.n_outputs = self.num_actions
        if self.func_approx is None:
            self.func_approx = self.init_mlp(self.n_inputs, self.num_actions)
        # set up training stuff
        self.optimizer = torch.optim.Adam(self.func_approx.parameters(), lr=self.learning_rate)
        self.base_loss_fn = torch.nn.MSELoss().to(self.device)
        self.update_target_net(reinit=True)

    def init_mlp(self, input_len, output_len):
        """ Initialize an MLP as a function approximator """

        model = torch.nn.Sequential(
            torch.nn.Linear(input_len, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_len)
        )

        model.to(self.device)
        
        return model
    
    def update_target_net(self, reinit=False):
        if reinit:
            self.target_func_approx = self.init_mlp(self.n_inputs, self.num_actions)

        if self.update_target_type == 'hard':
            if self.step % self.update_timestep == 0:
                self.target_func_approx.load_state_dict(self.func_approx.state_dict())
                self.target_func_approx.to(self.device)
                self.target_func_approx.eval()
        elif self.update_target_type == 'soft':
            for target_param, param in zip(self.target_func_approx.parameters(), self.func_approx.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        elif self.update_target_type == 'none':
            return

    def calc_action_values(self, inputs):
        """ Return Q(s,a) for a given state s:=inputs,
            for an MLP, forward pass the inputs and get a prediction"""
        self.func_approx.eval()
        return self.func_approx(inputs) # [0] to get rid of batch dimension=1

    def random_action(self, probs=None):
        """ Choose a random action (random arm to pull), with probabilities probs
            If probs is None, then each choice is equally likely. """
        if sum(probs) != 1.0:
            probs = probs / sum(probs)
        return torch.distributions.Categorical(probs=probs).sample().item()

    def e_greedy(self, Q_values):
        """ Greedily choose the action with highest est. action-value,
            or choose another random one with probability epsilon """
        n_actions = len(Q_values)

        if random.random() < self.epsilon:
            return torch.randint(0, n_actions, (1,)).item()
        else:
            return torch.argmax(Q_values).item()

    def e_greedy_policy(self, Q_values):
        """ Return the epsilon-greedy policy for a given state in the form of a tensor of probabilities """
        
        num_actions = Q_values.size(0)
        policy = torch.ones(num_actions) * (self.epsilon / num_actions)
        best_action = torch.argmax(Q_values).item()
        policy[best_action] += 1.0 - self.epsilon
        
        return policy

    def softmax_policy(self, Q_values):
        """ Return the softmax policy for a given state in the form of a tensor of probabilities """
        return torch.exp(Q_values) / torch.sum(torch.exp(Q_values))

    def act(self, states):
        """ Choose an action for each agent, given the current state """

        action = self.get_decision(states)

        self.prev_action = action
        self.prev_state = states
        
        return action

    def get_shielded_action(self, x, base_actions):
        sensor_values = x
        if self.shield is not None:
            sensor_values = self.shield.get_sensor_values(x)

        # sensor_values = sensor_values.to("cpu")
        # base_actions = base_actions.to("cpu")

        self.debug_info = {"sensor_value": sensor_values, "base_policy": base_actions}

        actions = None
        safety = -1
        if self.shield is None:
            self.debug_info["shielded_policy"] = base_actions
            actions = base_actions

        elif self.shield.differentiable:  # PLPG
            # compute the shielded policy
            actions = self.shield.get_shielded_policy(base_actions.unsqueeze(0), sensor_values.unsqueeze(0))
            # shielded_policy = Categorical(probs=actions)
            safety = self.shield.get_policy_safety(sensor_values.unsqueeze(0), base_actions.unsqueeze(0))

            self.debug_info["shielded_policy"] = actions
            self.debug_info["safety"] = safety
            self.debug_info["action_safety"] = self.shield.get_action_safeties(sensor_values.unsqueeze(0))

        else:  # VSRL
            with torch.no_grad():
                actions = self.shield.get_shielded_policy_vsrl(
                    base_actions.unsqueeze(0), sensor_values.unsqueeze(0)
                )
                shielded_policy = Categorical(probs=actions)

                safety = self.shield.get_policy_safety(sensor_values.unsqueeze(0), base_actions.unsqueeze(0))
                
                # get the most probable action of the shielded policy if we want to use a deterministic policy,
                # otherwuse, sample an action
                # if deterministic:
                #     actions = torch.argmax(shielded_policy.probs, dim=1)
                # else:
                #     actions = shielded_policy.sample()

                # log_prob = distribution.log_prob(actions)
                self.debug_info["shielded_policy"] = shielded_policy.probs
                self.debug_info["safety"] = safety
                self.debug_info["action_safety"] = self.shield.get_action_safeties(sensor_values.unsqueeze(0))

                # return (actions, values, log_prob)
        
        self.debug_info_history.append(self.debug_info)
        return actions.to(self.device)

    def get_decision(self, state):
        """ Wrapper function for returning an action for a given state,
            and also includes necessary updates to memory and epsilon """
        
        state = torch.Tensor(state).to(self.device)

        # Q_values = self.calc_action_values(state.unsqueeze(0)).squeeze(0).detach()
        Q_values_prob, Q_values = self.get_action_probs(state, return_action_values=True)
        action = self.random_action(Q_values_prob)

        if self.training:
            # update epsilon
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

            self._temp_history = [None]*6
            self._temp_history[STATE] = state
            self._temp_history[Q_VALS] = Q_values
            self._temp_history[ACTION] = action
            self._temp_history[REWARD] = None
            self._temp_history[TERMINAL] = None
            self._temp_history[ACTION_DIST] = Q_values_prob

        return self.controls[action]

    def update_reward(self, reward, terminal=False):
        """ Record the current reward and whether or not the state is terminal. """
        if self.training:
            
            self.history.append([None]*6)
            self.history[-1][STATE] = self._temp_history[STATE]
            self.history[-1][Q_VALS] = self._temp_history[Q_VALS]
            self.history[-1][ACTION] = self._temp_history[ACTION] 
            self.history[-1][REWARD] = reward
            self.history[-1][TERMINAL] = terminal
            self.history[-1][ACTION_DIST] = self._temp_history[ACTION_DIST]
            # forget the oldest memories to make room for new ones
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # update the target network
            self.update_target_net() 
            
            self.step += 1
        
        self.reward = reward

        if not self.eval_mode and self.training and self.prev_state is not None:
            for _ in range(self.epochs):
                self.train_model()
    
    def get_safety_loss(self, sensor_values_batch, base_actions_batch):
        ####### Safety loss ###########################################
        if self.shield is None:
            # no shield
            safety_loss = torch.tensor([0]*len(sensor_values_batch)).to(self.device)
        else:
            policy_safeties = self.shield.get_policy_safety(sensor_values_batch, base_actions_batch)
            policy_safeties = policy_safeties.flatten()
            safety_loss = -torch.log(policy_safeties)
            safety_loss = torch.mean(safety_loss)

        loss = self.alpha * safety_loss
        return loss
        ###############################################################
    
    def normalize_Q_values(self, Q_values):
        if not self.eval_mode:
            if self.explore_policy == 'softmax':
                Q_values_norm = self.softmax_policy(Q_values)
            elif self.explore_policy == 'e_greedy':
                Q_values_norm = self.e_greedy_policy(Q_values)
            else:
                raise ValueError(f"Invalid explore_policy {self.explore_policy}")
        else:
            if self.eval_policy == 'softmax':
                Q_values_norm = self.softmax_policy(Q_values)
            elif self.eval_policy == 'greedy':
                Q_values_norm = torch.zeros(self.num_actions)
                Q_values_norm[torch.argmax(Q_values)] = 1.0
            else:
                raise ValueError(f"Invalid eval_policy {self.eval_policy}")
    
        return Q_values_norm.to(self.device, dtype=torch.float32)
    
    def get_action_probs(self, state, return_action_values=False):
        with torch.no_grad():
            state = torch.tensor(state).to(self.device)
            Q_values = self.calc_action_values(state.unsqueeze(0)).squeeze(0).detach()
            Q_values_norm = self.normalize_Q_values(Q_values)
        
        if self.eval_mode and self.eval_policy == 'greedy': # shield breaks if there's no safe actions
            eps = self.epsilon
            self.epsilon = 0.001
            Q_values_norm = self.e_greedy_policy(Q_values).to(self.device)
            self.epsilon = eps
 
        shielded_policy = self.get_shielded_action(state, Q_values_norm).squeeze(0).to(self.device)
        
        if return_action_values:
            return shielded_policy, Q_values
        
        return shielded_policy

    def train_model(self):
        """ Train the MLP function approximator using experience replay """

        if not self.training:
            return

        if len(self.history) <= self.batch_size:
            return
    
        # Uniformly sample batch indices
        batch_indices = torch.randint(0, len(self.history) - 1, (self.batch_size,), device=self.device)

        # Extract batch data
        batch = [self.history[idx] for idx in batch_indices.cpu().numpy()]
        next_batch = [self.history[idx + 1] for idx in batch_indices.cpu().numpy()]

        cur_states = torch.stack([item[STATE] for item in batch]).to(self.device)
        cur_actions = torch.tensor([item[ACTION] for item in batch], device=self.device)
        cur_Q_vals = torch.stack([item[Q_VALS] for item in batch]).to(self.device)
        rewards = torch.tensor([item[REWARD] for item in batch], device=self.device)
        terminals = torch.tensor([item[TERMINAL] for item in batch], dtype=torch.bool, device=self.device)
        next_states = torch.stack([item[STATE] for item in next_batch]).to(self.device)

        # Compute next Q-values for all next states
        next_Q_vals = self.target_func_approx(next_states).detach()

        # Compute targets
        if self.on_policy:
            next_actions = torch.tensor([item[ACTION] for item in next_batch], device=self.device)
            next_Q_values_selected = next_Q_vals[torch.arange(next_Q_vals.size(0)), next_actions]
        else:
            next_Q_values_selected = next_Q_vals.max(dim=1).values

        # Update targets based on the SARSA rule
        updated_targets = cur_Q_vals.clone()
        updated_targets[torch.arange(cur_Q_vals.size(0)), cur_actions] = rewards + (~terminals) * (self.gamma * next_Q_values_selected)

        # Train using X_train and y_train
        X_train = cur_states
        y_train = updated_targets
        base_loss = self.base_loss_fn(self.func_approx(X_train), y_train)

        cur_states = torch.stack([self.history[int(i)][STATE] for i in batch_indices]).to(self.device).float()
        cur_action_dist = torch.stack([self.history[int(i)][ACTION_DIST] for i in batch_indices]).to(self.device).float()

        safety_augmentations =  self.get_safety_loss(cur_states, cur_action_dist)
        safety_loss = torch.mean(safety_augmentations).to(self.device)
        loss = base_loss + safety_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        losses = {"base_loss": base_loss, "safety_loss": safety_loss}
        # print(losses)
        self.loss_info.append(losses)

    def save(self, filename):
        """ Save a copy of the current model to file """
        torch.save(self.func_approx.state_dict(), filename)
        self.saved_model_name = filename

    def save_details(self, filename):
        """ Save a text copy of the hyperparams and other details of this agent object """
        with open(filename+".txt", 'w') as f:
            deets =  { k:v for k,v in vars(self).items() if k not in ['history','func_approx'] }
            f.write(str(deets))

        if self.save_debug_info:
            with open(filename + "_debug_info.txt", 'w') as f:
                f.write(str(self.debug_info_history))

    def ignore_shield(self):
        # why is this here again?
        return
        self.shield = None

    def set_eval_mode(self, bool_val):
        """ Set the agent to evaluation mode """
        self.eval_mode = bool_val
        self.training = not bool_val

    def load(self, filename=None):
        """ Load model for training or optional exploitative deployment """
        if filename is not None:
            self.saved_model_name = filename
        self.func_approx = self.init_mlp(self.n_inputs, self.num_actions)
        self.func_approx.load_state_dict(torch.load(self.saved_model_name))
        self.func_approx.to(self.device)

        self.target_func_approx = self.init_mlp(self.n_inputs, self.num_actions)
        self.target_func_approx.load_state_dict(self.func_approx.state_dict())
        self.target_func_approx.eval()

    def begin_episode(self):
        self.prev_states = None
        # self.prev_actions = None
        # self.rewards = None
        # self.epsilon = self.epsilon_start
        pass
    
    def get_params(self):
        return self.func_approx
    
    def set_params(self, params):   
        self.func_approx = params