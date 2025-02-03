import os
import datetime
import pickle as pk

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

import action_wrappers
import sensor_wrappers
from shield_selector import ShieldSelector

from algos import *
from env_selection import ALL_ENVS, ALL_ENVS_ARGS
from config import config

from run_episode import run_episode, eval_episode

np.random.seed(config.seed)
torch.manual_seed(config.seed)

device = torch.device(config.device)

system = os.name

# cur_time = time.time()
now = datetime.datetime.now()
cur_time = now.strftime("%Y-%m-%d_%H%M%S")

print("[INFO] Current time:", cur_time)

# set up environment
max_training_episodes = config.max_eps
max_cycles = config.max_cycles

total_cycles = max_training_episodes * max_cycles
print(f"[INFO] Training for {max_training_episodes} episodes of {max_cycles} cycles each, totalling {total_cycles} cycles.")

env_creator_func = ALL_ENVS[config.env]
env_creator_args = ALL_ENVS_ARGS[config.env]
env_creator_args.update({"max_cycles": max_cycles})
env = env_creator_func(render_mode=None, **env_creator_args)
env.reset()

env_name = env.metadata["name"]
n_eval = config.n_eval

n_discrete_actions = env.action_space(env.possible_agents[0]).n
if hasattr(env.observation_space(env.possible_agents[0]), "shape") and len(env.observation_space(env.possible_agents[0]).shape) > 0: 
    observation_space = env.observation_space(env.possible_agents[0]).shape[0]  # for box spaces with shape
else: 
    observation_space = env.observation_space(env.possible_agents[0]).n         # for discrete spaces?
print(f"[INFO] Observation space: {observation_space}, Action space: {n_discrete_actions}")

action_wrapper = action_wrappers.get_wrapper(env_name, n_discrete_actions, device)
sensor_wrapper = sensor_wrappers.get_wrapper(env_name)(env, observation_space, device)

shield_selector = ShieldSelector(env_name=env_name, 
                                 n_actions=action_wrapper.num_actions, 
                                 n_sensors=sensor_wrapper.num_sensors,
                                 filename=config.shield_file,
                                 version=config.shield_version
                                 )

sh_params = {
    "config_folder": shield_selector.base_dir,
    "num_sensors": sensor_wrapper.num_sensors,
    "num_actions": action_wrapper.num_actions,
    "differentiable": config.shield_diff,
    "vsrl_eps": config.shield_eps,
    "shield_program": shield_selector.file,
    "observation_type": "ground truth",
    "get_sensor_value_ground_truth": sensor_wrapper,
    "device": device
}

ppo_params = ["update_timestep", "train_epochs", "gamma", "eps_clip", "lr_actor", "lr_critic", "vf_coef", "entropy_coef", "device"]
dqn_params = ["update_timestep", "train_epochs", "gamma", "buffer_size", "batch_size", "lr", "eps_decay", "eps_min", "tau", "target_update_type", "explore_policy", "eval_policy", "on_policy", "device"]
extracted_ppo = {k: v for k, v in vars(config).items() if k in ppo_params}
extracted_dqn = {k: v for k, v in vars(config).items() if k in dqn_params}
all_algo_params = {k: v for k, v in vars(config).items() if k in ppo_params or k in dqn_params}
alpha = config.shield_alpha


############################################ ALGO SELECTION ############################################

algo_name = config.algo

if algo_name not in ALL_ALGORITHMS:
    raise ValueError(f"Algorithm '{algo_name}' not found. Available styles are {list(ALL_ALGORITHMS.keys())}")

algo = ALL_ALGORITHMS[algo_name](env=env, 
                                 observation_space=observation_space,
                                 n_discrete_actions=n_discrete_actions,
                                 action_wrapper=action_wrapper,
                                 sensor_wrapper=sensor_wrapper,
                                 sh_params=sh_params,
                                 algorithm_params=all_algo_params,
                                 alpha=alpha,
                                 shielded_ratio=config.shielded_ratio,
                                 device=device
                                 )

############################################ SAFETY CALC ############################################

if config.shield_eval_version == -1:
    config.shield_eval_version = config.shield_version
    
shield_selector_calc = ShieldSelector(env_name=env_name, 
                                 n_actions=action_wrapper.num_actions, 
                                 n_sensors=sensor_wrapper.num_sensors,
                                 filename=config.shield_file,
                                 version=config.shield_eval_version
                                 )
sh_params_calc = sh_params.copy()
sh_params_calc.update({"shield_program": shield_selector_calc.file})

safety_calc = SafetyCalculator(sh_params_calc)
# safety_calc = None

############################################ SAVE CONFIG ############################################
config_dict = vars(config)
config_dict.update(env_creator_args)

# safe config dict to json file
import json
if not os.path.exists(f"histories/configs/{env_name}/{algo_name}/{cur_time}.json"):
    os.makedirs(f"histories/configs/{env_name}/{algo_name}", exist_ok=True)
with open(f"histories/configs/{env_name}/{algo_name}/{cur_time}.json", "w") as f:
    s = json.dumps(config_dict, indent=4)
    f.write(s)

with open("histories/configs/run_history.csv", "a") as f:
    f.write(f"{cur_time}, {env_name}, {algo_name}_{cur_time}, \n")

############################################ TRAINING ############################################

reward_hists = []
eval_hists = []
eval_safeties = []
eval_episodes = []
mode = "online" if config.use_wandb else "disabled"
wandb.init(project=f"{system}_{env_name}", name=f"{algo_name}_{cur_time}", config=config_dict, mode=mode)

ep=0
try:
    for _ in range(max_training_episodes):
        reward_hist = run_episode(env, algo, max_cycles, ep)
        reward_hists.append(reward_hist)

        if ep % config.eval_every == 0 or ep == max_training_episodes-1:
            eval_episodes.append(ep)
            eval_reward_hists = []
            eval_safety_hists = []
            for _ in range(n_eval):
                eval_reward_hist, eval_safety_hist = eval_episode(env, algo, max_cycles, safety_calculator=safety_calc)
                eval_reward_hists.append(eval_reward_hist)
                eval_safety_hists.append(eval_safety_hist)
            eval_hists.append(eval_reward_hists)
            eval_safeties.append(eval_safety_hists)
            if "eval_funcs" in dir(env):
                for eval_func in env.eval_funcs:
                    eval_func(env, algo, ep)

            algo.save(f"models/{env_name}/{algo_name}_{cur_time}/ep{ep}")

        ep+=1

except KeyboardInterrupt:
    print("Training interrupted, saving model.")
    algo.save(f"models/{env_name}/{algo_name}_{cur_time}/ep{ep}")

wandb.finish()
env.close()



############################################ POST TRAINING ############################################

############################################ SAVE DATA ############################################
# for r, reward_hist in enumerate(reward_hists):
#     print(f"Episode {r} : ", end="")    
#     print({a:np.sum(reward_hist[a]) for a in reward_hist.keys()})
#     print("Total reward: ", np.sum([np.sum(reward_hist[a]) for a in reward_hist.keys()]))

# save hists as pickle
if not os.path.exists(f"histories/{env_name}/{algo_name}/{cur_time}_train.pk"):
    os.makedirs(f"histories/{env_name}/{algo_name}", exist_ok=True)
with open(f"histories/{env_name}/{algo_name}/{cur_time}_train.pk", "wb") as f:
    pk.dump(reward_hists, f)

if not os.path.exists(f"histories/{env_name}/{algo_name}/{cur_time}_eval.pk"):
    os.makedirs(f"histories/{env_name}/{algo_name}", exist_ok=True)
with open(f"histories/{env_name}/{algo_name}/{cur_time}_eval.pk", "wb") as f:
    pk.dump(eval_hists, f)

if not os.path.exists(f"histories/{env_name}/{algo_name}/{cur_time}_safety.pk"):
    os.makedirs(f"histories/{env_name}/{algo_name}", exist_ok=True)
with open(f"histories/{env_name}/{algo_name}/{cur_time}_safety.pk", "wb") as f:
    pk.dump(eval_safeties, f)

if not os.path.exists(f"histories/{env_name}/{algo_name}/{cur_time}_eval_eps.pk"):
    os.makedirs(f"histories/{env_name}/{algo_name}", exist_ok=True)
with open(f"histories/{env_name}/{algo_name}/{cur_time}_eval_eps.pk", "wb") as f:
    pk.dump(eval_episodes, f)

print("Training complete. Fileref:", cur_time)
print("Runtime:", datetime.datetime.now() - now)


############################################ PLOT MEAN REWARDS ############################################


# plot mean rewards per episode
agent_rewards = [[np.mean(reward_hist[agent]) for reward_hist in reward_hists] for agent in reward_hists[0].keys()]
for a, agent in enumerate(algo.agents.keys()):
    plt.plot(agent_rewards[a], label=agent)

plt.plot([np.mean([np.mean(reward_hist[agent]) for agent in reward_hist.keys()]) for reward_hist in reward_hists], label="mean", color="black", linestyle="--")

plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title(f"{algo_name} on {env_name} (training)")

plt.legend()
if not os.path.exists(f"plots/{env_name}/{algo_name}"):
    os.makedirs(f"plots/{env_name}/{algo_name}")
plt.grid(True)
plt.savefig(f"plots/{env_name}/{algo_name}/{cur_time}_train_mean.png")
# plt.show()
plt.clf()

############################################ PLOT SUM REWARDS ############################################

# plot sum of rewards per episode
agent_rewards = [[np.sum(reward_hist[agent]) for reward_hist in reward_hists] for agent in reward_hists[0].keys()]
for a, agent in enumerate(algo.agents.keys()):
    plt.plot(agent_rewards[a], label=agent)

plt.plot([np.mean([np.sum(reward_hist[agent]) for agent in reward_hist.keys()]) for reward_hist in reward_hists], label="mean", color="black", linestyle="--")

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title(f"{algo_name} on {env_name} (training)")

plt.legend()
if not os.path.exists(f"plots/{env_name}/{algo_name}"):
    os.makedirs(f"plots/{env_name}/{algo_name}")
plt.grid(True)
plt.savefig(f"plots/{env_name}/{algo_name}/{cur_time}_train_total.png")
# plt.show()
plt.clf()

############################################ PLOT EVAL MEAN REWARDS ############################################

# compute eval means and stds
eval_means = {}
eval_stds = {}

for a, agent in enumerate(algo.agents.keys()):
    eval_means[agent] = []
    eval_stds[agent] = []
    for eval_hist in eval_hists:
        eval_means[agent].append(np.mean([np.mean(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))
        eval_stds[agent].append(np.std([np.mean(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))

# compute overall mean and std
eval_means["mean"] = []
eval_stds["mean"] = []
for eval_hist in eval_hists:
    eval_means["mean"].append(np.mean([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))
    eval_stds["mean"].append(np.std([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))

# plot eval info
for a, agent in enumerate(algo.agents.keys()):
    plt.errorbar(eval_episodes, eval_means[agent], yerr=eval_stds[agent], label=f"{agent} mean", capsize=5, marker="x")

plt.errorbar(eval_episodes, eval_means["mean"], yerr=eval_stds["mean"], label="mean", color="black", linestyle="--", capsize=5, marker="x")

# plt.xticks(eval_episodes)
plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title(f"{algo_name} on {env_name} (evaluation)")
plt.grid(True)
plt.legend()
plt.savefig(f"plots/{env_name}/{algo_name}/{cur_time}_eval_mean.png")

plt.clf()

############################################ PLOT EVAL SUM REWARDS ############################################

# compute eval sums and stds
eval_sums = {}
eval_stds = {}

for a, agent in enumerate(algo.agents.keys()):
    eval_sums[agent] = []
    eval_stds[agent] = []
    for eval_hist in eval_hists:
        eval_sums[agent].append(np.mean([np.sum(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))
        eval_stds[agent].append(np.std([np.sum(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))

# compute overall mean and std
eval_sums["mean"] = []
eval_stds["mean"] = []
for eval_hist in eval_hists:
    eval_sums["mean"].append(np.mean([np.mean([np.sum(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))
    eval_stds["mean"].append(np.std([np.mean([np.sum(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))

# plot eval info
for a, agent in enumerate(algo.agents.keys()):
    plt.errorbar(eval_episodes, eval_sums[agent], yerr=eval_stds[agent], label=f"{agent} total", capsize=5, marker="x")

plt.errorbar(eval_episodes, eval_sums["mean"], yerr=eval_stds["mean"], label="mean", color="black", linestyle="--", capsize=5, marker="x")

# plt.xticks(eval_episodes)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title(f"{algo_name} on {env_name} (evaluation)")
plt.grid(True)
plt.legend()
plt.savefig(f"plots/{env_name}/{algo_name}/{cur_time}_eval_total.png")

plt.clf()

############################################ PLOT EVAL MEAN SAFETY ############################################

if safety_calc is not None:
    # plot safety info
    # compute eval means and stds
    eval_means = {}
    eval_stds = {}

    for a, agent in enumerate(algo.agents.keys()):
        eval_means[agent] = []
        eval_stds[agent] = []
        for eval_hist in eval_safeties:
            eval_means[agent].append(np.mean([np.mean(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))
            eval_stds[agent].append(np.std([np.mean(eval_reward_hist[agent]) for eval_reward_hist in eval_hist]))

    # compute overall mean and std
    eval_means["mean"] = []
    eval_stds["mean"] = []
    for eval_hist in eval_safeties:
        eval_means["mean"].append(np.mean([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))
        eval_stds["mean"].append(np.std([np.mean([np.mean(eval_reward_hist[agent]) for agent in eval_reward_hist.keys()]) for eval_reward_hist in eval_hist]))

    # plot eval info
    for a, agent in enumerate(algo.agents.keys()):
        plt.errorbar(eval_episodes, eval_means[agent], yerr=eval_stds[agent], label=f"{agent} mean", capsize=5, marker="x")

    plt.errorbar(eval_episodes, eval_means["mean"], yerr=eval_stds["mean"], label="mean", color="black", linestyle="--", capsize=5, marker="x")

    # plt.xticks(eval_episodes)
    plt.xlabel("Episode")
    plt.ylabel("Mean Safety")
    plt.title(f"{algo_name} on {env_name} (safety evaluations)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"plots/{env_name}/{algo_name}/{cur_time}_safeties.png")

    # plt.show()
exit()

############################################ OFFLINE EVALUATION ############################################
# set up environment
env = env_creator_func(render_mode=None, **env_creator_args)
env.reset()
# evaluation episodes
del algo
algo = ALL_ALGORITHMS[algo_name](env=env, 
                                 observation_space=observation_space,
                                 n_discrete_actions=n_discrete_actions,
                                 action_wrapper=action_wrapper,
                                 sensor_wrapper=sensor_wrapper,
                                 sh_params=sh_params,
                                 algorithm_params=all_algo_params,
                                 alpha=alpha
                                 )
algo.load(f"models/{env_name}/{algo_name}_{cur_time}/ep{max_training_episodes-1}")
# algo.update_env(env)
reward_hists = []
for ep in range(5):
    reward_hist = eval_episode(env, algo, max_cycles)
    reward_hists.append(reward_hist)

for reward_hist in reward_hists:
    print({a:np.sum(reward_hist[a]) for a in reward_hist.keys()})

env.close() 