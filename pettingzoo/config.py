from algos import *
from env_selection import ALL_ENVS
import json

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--config_file",
                    type=str,
                    default="",
                    help="Config file to use."
                    )

parser.add_argument("--use_wandb",
                    action="store_true",
                    help="Use wandb for logging."
                    )

parser.add_argument("--wandb_project_prefix",
                    type=str,
                    default="",
                    help="Prefix for wandb project name."
                    )

parser.add_argument("--device",
                    type=str,
                    default="cpu",
                    help="Device to use for training."
                    )

# env and algo params
parser.add_argument("--algo",
                    type=str,
                    default="IQL",
                    help=f"Training style to use. Options are {ALL_ALGORITHMS.keys()}"
                    )

parser.add_argument("--env",
                    type=str,
                    default="simple_pd_v0",
                    help=f"Environment to use. Options are {ALL_ENVS.keys()}"
                    )

parser.add_argument("--max_cycles",
                    type=int,
                    default=25,
                    help="Number of cycles to run each environment episode for."
                    )

parser.add_argument("--max_eps",
                    type=int,
                    default=500,
                    help="Number of episodes to train for."
                    )

parser.add_argument("--eval_every",
                    type=int,
                    default=50,
                    help="Number of episodes between evaluations."
                    )

parser.add_argument("--n_eval",
                    type=int,
                    default=5,
                    help="Number of episodes to evaluate for each evaluation."
                    )

parser.add_argument("--seed",
                    type=int,
                    default=-1,
                    help="Seed for reproducibility (-1 for auto)."
                    )

parser.add_argument("--shield_alpha",
                    type=float,
                    default=1.0,
                    help="Alpha value for shield."
                    )

parser.add_argument("--shield_file",
                    type=str,
                    default="default",
                    help="Shield file to use."
                    )

parser.add_argument("--shield_version",
                    type=int,
                    default=0,
                    help="Version of shield to use for a given env."
                    )

parser.add_argument("--shield_eval_version",
                    type=int,
                    default=-1,
                    help="Version of shield to use to evaluate agents for a given env (default = shield_version)."
                    )

parser.add_argument("--shielded_ratio",
                    type=float,
                    default=1.0,
                    help="Ratio of shielded agents to total (minimum)."
                    )

parser.add_argument("--shield_diff",
                    action="store_true",
                    help="Use differentiable shield (PLS vs VSRL)."
                    )

parser.add_argument("--shield_eps",
                    type=float,
                    default=0.1,
                    help="Epsilon value for VSRL shield."
                    )

# Model params, common to all algorithms
parser.add_argument("--update_timestep",
                    type=int,
                    default=100,
                    help="Update policy (PPO) / target network (DQN) every n timesteps."
                    )

parser.add_argument("--train_epochs",                                     
                    type=int,
                    default=10,
                    help="Train policy/Q network for K epochs in one PPO/DQN update."
                    )

parser.add_argument("--gamma",                                      
                    type=float,
                    default=0.99,
                    help="Discount factor."
                    )

# PPO Params
parser.add_argument("--eps_clip",
                    type=float,
                    default=0.1,
                    help="(PPO) Clip parameter for PPO."
                    )


parser.add_argument("--lr_actor",
                    type=float,
                    default=0.001,
                    help="(PPO) Learning rate for actor network."
                    )

parser.add_argument("--lr_critic",
                    type=float,
                    default=0.001,
                    help="(PPO) Learning rate for critic network."
                    )

parser.add_argument("--vf_coef",
                    type=float,
                    default=0.5,
                    help="Coefficient for value function loss."
                    )

parser.add_argument("--entropy_coef",
                    type=float,
                    default=0.01,
                    help="Coefficient for entropy loss."
                    )

# DQN Params
parser.add_argument("--buffer_size",
                    type=int,
                    default=1000,
                    help="(DQN) Size of the replay buffer."
                    )

parser.add_argument("--batch_size",
                    type=int,
                    default=64,
                    help="(DQN) Batch size for training."
                    )

parser.add_argument("--lr",
                    type=float,
                    default=0.001,
                    help="(DQN) Learning rate for the Q network."
                    )

parser.add_argument("--tau",
                    type=float,
                    default=0.01,
                    help="(DQN) Soft update parameter."
                    )

parser.add_argument("--target_update_type",
                    type=str,
                    default="soft",
                    help="(DQN) Type of update to use. Options are 'soft' and 'hard' and 'none' (no target network)."
                    )

parser.add_argument("--eps_min",
                    type=float,
                    default=0.05,
                    help="(DQN) Min epsilon for epsilon greedy policy."
                    )

parser.add_argument("--eps_decay",
                    type=float,
                    default=0.995,
                    help="(DQN) Decay rate for epsilon."
                    )

parser.add_argument("--explore_policy",
                    type=str,
                    default="e_greedy",
                    help="Exploration strategy to use. Options are 'e_greedy' and 'softmax'."
                    )

parser.add_argument("--eval_policy",
                    type=str,
                    default="greedy",
                    help="Exploitation strategy to use. Options are 'greedy' and 'softmax'."
                    )

parser.add_argument("--on_policy",
                    action="store_true",
                    help="Use on-policy updates for DQN."
                    )

config = parser.parse_args()

if config.config_file:
    print(f"[INFO] Using config file {config.config_file}")
    # json config file
    with open(config.config_file) as f:
        data = json.load(f)
        for key in data:
            setattr(config, key, data[key])