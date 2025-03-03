import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from pls.shields.shields import Shield

# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCriticShielded(nn.Module):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 shield_params=None,
                 get_sensor_value_ground_truth=None,
                 shield=None
                 ):
        super(ActorCriticShielded, self).__init__()

        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1)
                    )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )

        # shield
        self.get_sensor_value_ground_truth = get_sensor_value_ground_truth
        # agents can share a shield if a shield is passed in
        assert shield_params is None or shield is None, "Cannot pass in both shield_params and shield"
        self.shield = None
        if shield_params is not None:
            self.shield = Shield(
                **shield_params,
            )
        elif shield is not None:
            self.shield = shield
        elif shield_params is None and shield is None:
            self.shield = None
        
    def forward(self):
        raise NotImplementedError
    
    def get_actions(self, distribution, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return distribution.mode()
        return distribution.sample()
    
    def get_shielded_policy_batch_agnostic(self, base_actions, sensor_values):
        if self.shield is None:
            raise NotImplementedError("Shielded policy is only supported with a shield.")
    
        # single item batch
        if len(sensor_values.shape) == 1:
            return self.shield.get_shielded_policy(base_actions.unsqueeze(0), sensor_values.unsqueeze(0)).squeeze(0)
        # batch
        return self.shield.get_shielded_policy(base_actions, sensor_values)
    
    def act(self, state, deterministic=False):

        base_actions = self.actor(state)    # base_actions
        distribution = Categorical(base_actions)    # distribution

        if self.shield is None:
            sensor_values = self.get_sensor_value_ground_truth(state)
        else:
            sensor_values = self.shield.get_sensor_values(state)
        
        actions = None
        log_prob = None

        if self.shield is None:
            actions = self.get_actions(distribution, deterministic)
            log_prob = distribution.log_prob(actions)

        elif self.shield.differentiable:  # PLPG
            # compute the shielded policy
            if type(sensor_values) != torch.Tensor:
                sensor_values = torch.tensor(sensor_values, dtype=torch.float32).to(self.get_device())
            if type(base_actions) != torch.Tensor:
                base_actions = torch.tensor(base_actions, dtype=torch.float32).to(self.get_device())
            actions = self.get_shielded_policy_batch_agnostic(base_actions, sensor_values)
            shielded_policy = Categorical(probs=actions)

            # get the most probable action of the shielded policy if we want to use a deterministic policy,
            # otherwuse, sample an action
            if deterministic:
                actions = torch.argmax(shielded_policy.probs, dim=1)
            else:
                actions = shielded_policy.sample()

            log_prob = shielded_policy.log_prob(actions)

        else:  # VSRL
            with torch.no_grad():
                # TODO: batch agnostic version
                actions = self.shield.get_shielded_policy_vsrl(
                    base_actions, sensor_values
                )
                shielded_policy = Categorical(probs=actions)

                # get the most probable action of the shielded policy if we want to use a deterministic policy,
                # otherwuse, sample an action
                if deterministic:
                    actions = torch.argmax(shielded_policy.probs, dim=1)
                else:
                    actions = shielded_policy.sample()

                log_prob = distribution.log_prob(actions)

        state_val = self.critic(state)           # values

        return actions.detach(), log_prob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        distribution = Categorical(action_probs)
        # base_actions = self.get_actions(distribution, deterministic=False)
        base_actions = action_probs

        dist_entropy = None
        if self.shield is None or not self.shield.differentiable:
            sensor_values = self.get_sensor_value_ground_truth(state)
            log_prob = distribution.log_prob(action)
            dist_entropy = distribution.entropy()
            self.info = {"sensor_value": sensor_values, "base_policy": base_actions}

        elif self.shield.differentiable:  # PLPG
            sensor_values = self.shield.get_sensor_values(state)
            # compute the shielded policy
            shielded_actions = self.get_shielded_policy_batch_agnostic(base_actions, sensor_values)
            shielded_policy = Categorical(probs=shielded_actions)
            log_prob = shielded_policy.log_prob(action)
            dist_entropy = shielded_policy.entropy()
            self.info = {"sensor_value": sensor_values, "base_policy": base_actions}
            
        else:  # VSRL
            sensor_values = self.shield.get_sensor_values(state)
            log_prob = distribution.log_prob(action)
            dist_entropy = distribution.entropy()
            self.info = {"sensor_value": sensor_values, "base_policy": base_actions}

        state_values = self.critic(state)
        
        return log_prob, state_values, dist_entropy
    
    def get_device(self):
        # Check the device of the first parameter
        return next(self.parameters()).device

class PPOShielded:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 lr_actor, 
                 lr_critic, 
                 gamma, 
                 train_epochs, 
                 eps_clip,
                 update_timestep,
                 alpha=0,
                 policy_safety_params={},
                 policy_kw_args={},
                 vf_coef=0.5,
                 entropy_coef=0.01,
                 device="cpu",
                 **kwargs # made to be comptible with DQN
                 ):
        self.device = device
        print("[PPO INFO] Using device: ", self.device)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.train_epochs = train_epochs
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef

        self.buffer = RolloutBuffer()

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.policy_kw_args = policy_kw_args
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy = ActorCriticShielded(state_dim, action_dim, **policy_kw_args).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCriticShielded(state_dim, action_dim, **policy_kw_args).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.update_timestep = update_timestep

        self.MseLoss = nn.MSELoss()
        
        self.alpha = alpha
        self.policy_safety_calculater = Shield(**policy_safety_params)
        self.time_step = 0
        self.eval_mode = False

    def set_policy(self, policy):
        del self.policy
        del self.policy_old
        del self.optimizer

        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
            ])
        self.policy_old = ActorCriticShielded(self.state_dim, self.action_dim, **self.policy_kw_args).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict()).to(self.device)

    def set_policy_critic(self, critic):
        del self.optimizer

        self.policy.critic = critic.to(self.device)
        self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
            ])
        self.policy_old = ActorCriticShielded(self.state_dim, self.action_dim, **self.policy_kw_args).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def set_policy_actor(self, actor):
        del self.optimizer

        self.policy.actor = actor.to(self.device)
        self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
            ])
        self.policy_old = ActorCriticShielded(self.state_dim, self.action_dim, **self.policy_kw_args).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, state):
        with torch.no_grad():
            if type(state) != torch.Tensor:
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)
        
        if not self.eval_mode:
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

        return action.item()

    def get_action_probs(self, state):
        with torch.no_grad():
            if type(state) != torch.Tensor:
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action_probs = self.policy_old.actor(state)

        return action_probs

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        safety_losses = []
        # Optimize policy for K epochs
        for _ in range(self.train_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            ####### Safety loss ###########################################
            if vars(self.policy_safety_calculater) == {}:
                # no shield
                safety_loss = torch.tensor([0], device=self.device)
            else:
                policy_safeties = self.policy_safety_calculater.get_policy_safety(
                    self.policy.info["sensor_value"], self.policy.info["base_policy"]
                )
                policy_safeties = policy_safeties.flatten().to(self.device)
                safety_loss = -torch.log(policy_safeties)
                safety_loss = torch.mean(safety_loss)

            safety_losses.append(safety_loss.item())

            ###############################################################

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + self.vf_coef * self.MseLoss(state_values, rewards) - self.entropy_coef * dist_entropy + self.alpha * safety_loss
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.to(self.device)

        # clear buffer
        self.buffer.clear()
    
    def act(self, state):
        if type(state) != torch.Tensor:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        return self.select_action(state)

    def update_reward(self, reward, done):
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
        
        self.time_step += 1
        if self.time_step % self.update_timestep == 0:
                self.update()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def reset(self):
        pass

    def set_eval_mode(self, bool_val):
        """ Set the agent to evaluation mode """
        self.eval_mode = bool_val
        if bool_val:
            self.policy.eval()
            self.policy_old.eval()
        else:
            self.policy.train()
            self.policy_old.train()