import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

from pls.shields.shields import Shield

# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


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
                 has_continuous_action_space, 
                 action_std_init,
                 shield_params=None,
                 get_sensor_value_ground_truth=None,
                 shield=None
                 ):
        super(ActorCriticShielded, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            raise NotImplementedError("Only discrete action spaces are supported for now.")
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
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
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

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

        if self.has_continuous_action_space:
            raise NotImplementedError("Only discrete action spaces are supported for now.")
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            distribution = MultivariateNormal(action_mean, cov_mat)
        else:
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

        if self.has_continuous_action_space:
            raise NotImplementedError("Only discrete action spaces are supported for now.")
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
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
            
        # elif self.shield.differentiable:  # PLPG
        #     # compute the shielded policy
        #     actions = self.get_shielded_policy_batch_agnostic(base_actions, sensor_values)
        #     shielded_policy = Categorical(probs=actions)

        #     # get the most probable action of the shielded policy if we want to use a deterministic policy,
        #     # otherwuse, sample an action
        #     if deterministic:
        #         actions = torch.argmax(shielded_policy.probs, dim=1)
        #     else:
        #         actions = shielded_policy.sample()

        #     log_prob = shielded_policy.log_prob(actions)

        else:  # VSRL
            sensor_values = self.shield.get_sensor_values(state)
            log_prob = distribution.log_prob(action)
            dist_entropy = distribution.entropy()
            self.info = {"sensor_value": sensor_values, "base_policy": base_actions}

        state_values = self.critic(state)
        
        return log_prob, state_values, dist_entropy


class PPOShielded:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 lr_actor, 
                 lr_critic, 
                 gamma, 
                 K_epochs, 
                 eps_clip,
                 update_timestep,
                 has_continuous_action_space=False, 
                 action_std_init=0.6,
                 alpha=0,
                 policy_safety_params={},
                 policy_kw_args={}
                 ):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.policy_kw_args = policy_kw_args
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.has_continuous_action_space = has_continuous_action_space
        self.action_std_init = action_std_init
        self.policy = ActorCriticShielded(state_dim, action_dim, has_continuous_action_space, action_std_init, **policy_kw_args).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCriticShielded(state_dim, action_dim, has_continuous_action_space, action_std_init, **policy_kw_args).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.update_timestep = update_timestep

        self.MseLoss = nn.MSELoss()
        
        self.alpha = alpha
        self.policy_safety_calculater = Shield(**policy_safety_params)
        self.time_step = 0

    def set_policy(self, policy):
        del self.policy
        del self.policy_old
        del self.optimizer
        
        self.policy = policy.to(device)
        self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
            ])
        self.policy_old = ActorCriticShielded(self.state_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init, **self.policy_kw_args).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())


    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

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
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        safety_losses = []
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

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
                safety_loss = torch.Tensor([0])
            else:
                policy_safeties = self.policy_safety_calculater.get_policy_safety(
                    self.policy.info["sensor_value"], self.policy.info["base_policy"]
                )
                policy_safeties = policy_safeties.flatten()
                safety_loss = -torch.log(policy_safeties)
                safety_loss = torch.mean(safety_loss)

            safety_losses.append(safety_loss.item())

            ###############################################################

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy + self.alpha * safety_loss
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def act(self, state):
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

