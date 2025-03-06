# JSON Configuration File Documentation

This JSON configuration file provides settings for training and evaluation in reinforcement learning models. Each parameter is detailed below.

---

### General Configuration

- **`use_wandb`**: *(Boolean)*  
  Enable or disable `wandb` (Weights & Biases) for logging and experiment tracking.  
  - **Default**: `false`

- **`wandb_project`**: *(String)*
  Optional suffix to the wandb project name (the rest of the name is auto-generated).
  - **Default**: ``''`` *(empty string)*

- **`devide`**: *(String)*
  Device to use for training.
  - **Default**: `cpu`

- **`algo`**: *(String)*  
  Algorithm to be used for training.  
  - **Options**: `"IPPO"`, `"IQL"`, etc.

- **`env`**: *(String)*  
  Environment name for training the model.  
  - **Example**: `"simple_pd_v0"`

-- **`env_config`**: *(Str/Dict)*
  Environment configuration as a dictionary or a string containing a valid dictionary.
  - **Default**: `{}` (default env config in `env_selection.py`)

- **`max_cycles`**: *(Integer)*  
  Maximum number of cycles in each episode.  
  - **Default**: `25`

- **`max_eps`**: *(Integer)*  
  Maximum number of episodes for training.  
  - **Default**: `100`

- **`eval_every`**: *(Integer)*  
  Interval (in episodes) between evaluations.  
  - **Default**: `10`

- **`n_eval`**: *(Integer)*  
  Number of episodes for each evaluation.  
  - **Default**: `1`

- **`seed`**: *(Integer)*  
  Seed value for reproducibility of the training process.  
  - **Default**: `2`

---

### Shield Parameters

- **`shield_alpha`**: *(Float)*  
  Alpha value that influences the application of the shield.  
  - **Default**: `1.0`

- **`shield_file`**: *(String)*  
  Name of the shield configuration file to use.  
  - **Default**: `"default"`

- **`shield_version`**: *(Integer)*  
  Shield version for training.  
  - **Default**: `0`

- **`shield_eval_version`**: *(Integer)*  
  Shield version specifically for evaluation; defaults to `shield_version`.  
  - **Default**: `0`

- **`shielded_ratio`**: *(Float)*  
  Minimum ratio of shielded agents to total agents.  
  - **Default**: `1.0`

- **`shield_diff`**: *(Boolean)*  
  Specifies if a differentiable shield should be used.  
  - **Default**: `true`

- **`shield_eps`**: *(Float)*  
  Epsilon parameter for the VSRL shield.  
  - **Default**: `0.1`

---

### Model and Training Parameters (PPO/DQN)

- **`update_timestep`**: *(Integer)*  
  Timesteps between updates of the policy or target network.  
  - **Default**: `50`

- **`train_epochs`**: *(Integer)*  
  Number of epochs for each training update.  
  - **Default**: `10`

- **`gamma`**: *(Float)*  
  Discount factor for future rewards.  
  - **Default**: `0.99`

---

### PPO-Specific Parameters

- **`eps_clip`**: *(Float)*  
  Clipping parameter for PPO to restrict policy updates.  
  - **Default**: `0.1`

- **`lr_actor`**: *(Float)*  
  Learning rate for the actor network.  
  - **Default**: `0.001`

- **`lr_critic`**: *(Float)*  
  Learning rate for the critic network.  
  - **Default**: `0.001`

- **`vf_coef`**: *(Float)*
  Coefficient for value function loss.
  - **Default**: `0.5`

- **`entropy_coef`**: *(Float)*
  Coefficient for entropy loss.
  - **Default**: `0.01`

---

### DQN-Specific Parameters

- **`buffer_size`**: *(Integer)*  
  Size of the replay buffer in DQN.  
  - **Default**: `1000`

- **`batch_size`**: *(Integer)*  
  Batch size for training in DQN.  
  - **Default**: `64`

- **`lr`**: *(Float)*  
  Learning rate for the Q-network.  
  - **Default**: `0.001`

- **`tau`**: *(Float)*  
  Soft update rate for the target network.  
  - **Default**: `0.01`

- **`target_update_type`**: *(String)*  
  Type of target network update in DQN.
  - **Options**: `"soft"`, `"hard"` , `"none"`
  - **Default**: `"soft"`

- **`eps_min`**: *(Float)*  
  Minimum epsilon for exploration with an epsilon-greedy policy.  
  - **Default**: `0.05`

- **`eps_decay`**: *(Float)*  
  Decay rate for epsilon.  
  - **Default**: `0.995`

- **`explore_policy`**: *(String)*  
  Exploration strategy during training.  
  - **Options**: `"e_greedy"`, `"softmax"`  
  - **Default**: `"e_greedy"`

- **`eval_policy`**: *(String)*  
  Strategy for evaluation.  
  - **Options**: `"greedy"`, `"softmax"`  
  - **Default**: `"greedy"`

- **`on_policy`**: *(Boolean)*  
  Whether to use on-policy updates.  
  - **Default**: `false`