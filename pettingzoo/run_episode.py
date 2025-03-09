import numpy as np
import wandb
import time
from tqdm import trange
from algos import BaseMARLAlgo
import torch
import pprint 
pp = pprint.PrettyPrinter(indent=4)

############################################ RUN EPISODE ############################################

def run_episode(env, algo, max_cycles, ep=0):
    assert issubclass(type(algo), BaseMARLAlgo), "algo must be an instance of BaseMARLAlgo"
    reward_hist = {}
    observations, infos = env.reset()

    for step in trange(max_cycles, desc=f"Episode {ep}"):

        if len(env.agents) == 0:
            break
        
        actions = algo.act(observations)
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        algo.update_rewards(rewards, terminations, truncations)

        for agent in algo.agents.keys():
            if agent not in reward_hist:
                reward_hist[agent] = []
            reward_hist[agent].append(rewards[agent])
    
    # if the env has a env_update method, call it
    env_updates = {}
    if hasattr(env, "env_logging_info"):
        env_updates = env.env_logging_info(suffix="_training")

    update_dict = env_updates
    # mean reward per agent
    update_dict.update({f"mean_reward_{agent}": np.mean(reward_hist[agent]) for agent in reward_hist})
    # mean reward overall
    update_dict.update({"mean_reward": np.mean([np.mean(reward_hist[agent]) for agent in reward_hist])})
    # total reward per agent
    update_dict.update({f"total_reward_{agent}": np.sum(reward_hist[agent]) for agent in reward_hist})
    # total reward overall
    update_dict.update({"total_reward_mean": np.mean([np.sum(reward_hist[agent]) for agent in reward_hist])})
    # wandb.log(update_dict) # NOTE: wandb rate limit exceeded for sweep
    return reward_hist


############################################ EVALUATION ############################################

def eval_episode(env, algo, max_cycles, ep=0, safety_calculator=None, save_wandb=True):
    assert issubclass(type(algo), BaseMARLAlgo), "algo must be an instance of BaseMARLAlgo"
    reward_hist = {}
    safety_hist = {}
    observations, infos = env.reset()

    algo.eval(True)
    stag_chances = 0
    for step in range(max_cycles):

        if len(env.agents) == 0:
            break

        obs_old = observations
        actions = algo.act(observations)
        observations, rewards, terminations, truncations, infos = env.step(actions)

        if env.metadata["name"] == "markov_stag_hunt":
            for agent in algo.agents.keys():
                processed = algo.agents[agent].policy.get_sensor_value_ground_truth(torch.FloatTensor(obs_old[agent]))
                stag_chances += processed[-1] and processed[-2]

        for agent in algo.agents.keys():
            if agent not in reward_hist:
                reward_hist[agent] = []
            reward_hist[agent].append(rewards[agent])

        if safety_calculator is not None:
            safeties = safety_calculator(obs_old, algo)
            for agent in algo.agents.keys():
                if agent not in safety_hist:
                    safety_hist[agent] = []
                safety_hist[agent].append(safeties[agent].detach().cpu().numpy())

    # if the env has a env_update method, call it
    env_updates = {}
    if hasattr(env, "env_logging_info"):
        env_updates = env.env_logging_info(suffix="_eval")

    update_dict = env_updates
    # mean reward per agent
    update_dict.update({f"eval_mean_reward_{agent}": np.mean(reward_hist[agent]) for agent in reward_hist})
    # mean reward overall
    update_dict.update({"eval_mean_reward": np.mean([np.mean(reward_hist[agent]) for agent in reward_hist])})
    # total reward per agent
    update_dict.update({f"eval_total_reward_{agent}": np.sum(reward_hist[agent]) for agent in reward_hist})
    # total reward overall
    update_dict.update({"eval_total_reward_mean": np.mean([np.sum(reward_hist[agent]) for agent in reward_hist])})
    # mean safety per agent
    if safety_calculator is not None:
        update_dict.update({f"eval_mean_safety_{agent}": np.mean(safety_hist[agent]) for agent in safety_hist})
        # mean safety overall
        update_dict.update({"eval_mean_safety": np.mean([np.mean(safety_hist[agent]) for agent in safety_hist])})

    if save_wandb:
        wandb.log(update_dict)
    if env.metadata["name"] == "markov_stag_hunt":
        print("stag_chances", int(stag_chances.item()/2))
        pp.pprint(update_dict)

    algo.eval(False)

    
    return reward_hist, safety_hist