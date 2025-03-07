import os
import datetime
import pickle as pk
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

import wrappers.action_wrappers as action_wrappers
import wrappers.sensor_wrappers as sensor_wrappers
from shields.shield_selector import ShieldSelector

from algos import *
from env_selection import ALL_ENVS, ALL_ENVS_ARGS
from run_episode import run_episode, eval_episode
from config import config
from util import get_new_seed

def update_config_from_wandb():
    # Update config attributes from wandb.config if available.
    for key, value in wandb.config.items():
        setattr(config, key, value)
    print("Config updated from wandb:", vars(config))

def setup_experiment():
    # Set up seeds and device.
    config.seed = get_new_seed(config)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    
    # Set up timing and directories.
    now = datetime.datetime.now()
    cur_time = now.strftime("%Y-%m-%d_%H%M%S")
    print("[INFO] Current time:", cur_time)
    
    # Set up environment.
    max_training_episodes = config.max_eps
    max_cycles = config.max_cycles
    total_cycles = max_training_episodes * max_cycles
    print(f"[INFO] Training for {max_training_episodes} episodes of {max_cycles} cycles each, totalling {total_cycles} cycles.")
    
    env_creator_func = ALL_ENVS[config.env]
    env_creator_args = ALL_ENVS_ARGS[config.env]
    env_creator_args.update(config.env_config)
    env_creator_args.update({"max_cycles": max_cycles})
    env = env_creator_func(**env_creator_args)
    env.reset()

    # Setup wrappers, shields, and agent parameters.
    env_name = env.metadata["name"]
    n_eval = config.n_eval
    n_discrete_actions = env.action_space(env.possible_agents[0]).n
    if hasattr(env.observation_space(env.possible_agents[0]), "shape") and len(env.observation_space(env.possible_agents[0]).shape) > 0:
        observation_space = env.observation_space(env.possible_agents[0]).shape[0]
    else:
        observation_space = env.observation_space(env.possible_agents[0]).n
    print(f"[INFO] Observation space: {observation_space}, Action space: {n_discrete_actions}")

    action_wrapper = action_wrappers.get_wrapper(env_name, n_discrete_actions, device)
    sensor_wrapper = sensor_wrappers.get_wrapper(env_name)(env, observation_space, device)

    shield_selector = ShieldSelector(
        env_name=env_name, 
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
    
    # Extract algorithm parameters.
    ppo_params = ["update_timestep", "train_epochs", "gamma", "eps_clip", "lr_actor", "lr_critic", "vf_coef", "entropy_coef", "device"]
    dqn_params = ["update_timestep", "train_epochs", "gamma", "buffer_size", "batch_size", "lr", "eps_decay", "eps_min", "tau", "target_update_type", "explore_policy", "eval_policy", "on_policy", "device"]
    all_algo_params = {k: v for k, v in vars(config).items() if k in ppo_params or k in dqn_params}
    alpha = config.shield_alpha

    # Set up algorithm.
    algo_name = config.algo
    print(f"[INFO] Using algorithm: {algo_name}, with n_agents: {env.n_agents}")
    if algo_name not in ALL_ALGORITHMS:
        raise ValueError(f"Algorithm '{algo_name}' not found. Available styles are {list(ALL_ALGORITHMS.keys())}")
    algo = ALL_ALGORITHMS[algo_name](
        env=env, 
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
    
    # Set up safety calculator.
    if config.shield_eval_version == -1:
        config.shield_eval_version = config.shield_version
    shield_selector_calc = ShieldSelector(
        env_name=env_name, 
        n_actions=action_wrapper.num_actions, 
        n_sensors=sensor_wrapper.num_sensors,
        filename=config.shield_file,
        version=config.shield_eval_version
    )
    sh_params_calc = sh_params.copy()
    sh_params_calc.update({"shield_program": shield_selector_calc.file})
    from shields.safety_calculator import SafetyCalculator  # Ensure this import is valid.
    safety_calc = SafetyCalculator(sh_params_calc)
    
    # Save configuration.
    config_dict = vars(config)
    config_dict.update(env_creator_args)
    os.makedirs(f"histories/configs/{env_name}/{algo_name}", exist_ok=True)
    with open(f"histories/configs/{env_name}/{algo_name}/{cur_time}.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    with open("histories/configs/run_history.csv", "a") as f:
        f.write(f"{cur_time}, {env_name}, {algo_name}_{cur_time}, {config.seed}\n")
    
    return {
        "env": env,
        "algo": algo,
        "safety_calc": safety_calc,
        "cur_time": cur_time,
        "env_name": env_name,
        "max_training_episodes": config.max_eps,
        "max_cycles": config.max_cycles,
        "n_eval": config.n_eval,
        "config_dict": config_dict,
        "device": device
    }

def train_loop(setup_vars):
    env = setup_vars["env"]
    algo = setup_vars["algo"]
    safety_calc = setup_vars["safety_calc"]
    cur_time = setup_vars["cur_time"]
    env_name = setup_vars["env_name"]
    max_training_episodes = setup_vars["max_training_episodes"]
    max_cycles = setup_vars["max_cycles"]
    n_eval = setup_vars["n_eval"]
    
    reward_hists = []
    eval_hists = []
    eval_safeties = []
    eval_episodes = []
    
    ep = 0
    try:
        for _ in range(max_training_episodes):
            reward_hist = run_episode(env, algo, max_cycles, ep)
            reward_hists.append(reward_hist)
    
            if ep % config.eval_every == 0 or ep == max_training_episodes - 1:
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
                        eval_func(env=env, algo=algo, ep=ep, experiment_name=f"{config.algo}_{cur_time}")
    
                os.makedirs(f"models/{env_name}/{config.algo}_{cur_time}", exist_ok=True)
                algo.save(f"models/{env_name}/{config.algo}_{cur_time}/ep{ep}")
    
            ep += 1
    
    except KeyboardInterrupt:
        print("Training interrupted, saving model.")
        algo.save(f"models/{env_name}/{config.algo}_{cur_time}/ep{ep}")
    
    env.close()
    
    return {
        "reward_hists": reward_hists,
        "eval_hists": eval_hists,
        "eval_safeties": eval_safeties,
        "eval_episodes": eval_episodes
    }

def save_data(setup_vars, train_stats):
    env_name = setup_vars["env_name"]
    algo_name = config.algo
    cur_time = setup_vars["cur_time"]
    reward_hists = train_stats["reward_hists"]
    eval_hists = train_stats["eval_hists"]
    eval_safeties = train_stats["eval_safeties"]
    eval_episodes = train_stats["eval_episodes"]
    
    for suffix, data in zip(["_train.pk", "_eval.pk", "_safety.pk", "_eval_eps.pk"],
                              [reward_hists, eval_hists, eval_safeties, eval_episodes]):
        dir_path = f"histories/{env_name}/{algo_name}"
        os.makedirs(dir_path, exist_ok=True)
        with open(f"{dir_path}/{cur_time}{suffix}", "wb") as f:
            pk.dump(data, f)
    print("Data saved.")

def plot_results(setup_vars, train_stats):
    env_name = setup_vars["env_name"]
    algo_name = config.algo
    cur_time = setup_vars["cur_time"]
    reward_hists = train_stats["reward_hists"]
    
    agent_keys = list(reward_hists[0].keys())
    agent_rewards = [[np.mean(rh[agent]) for rh in reward_hists] for agent in agent_keys]
    for a, agent in enumerate(agent_keys):
        plt.plot(agent_rewards[a], label=agent)
    
    overall_mean = [np.mean([np.mean(rh[agent]) for agent in rh.keys()]) for rh in reward_hists]
    plt.plot(overall_mean, label="mean", color="black", linestyle="--")
    
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.title(f"{algo_name} on {env_name} (training)")
    plt.legend()
    plt.grid(True)
    
    plot_dir = f"plots/{env_name}/{algo_name}"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/{cur_time}_train_mean.png")
    plt.clf()
    print("Plots saved.")

def offline_evaluation(setup_vars):
    env_creator_func = ALL_ENVS[config.env]
    env_creator_args = ALL_ENVS_ARGS[config.env]
    env_creator_args.update(config.env_config)
    env_creator_args.update({"max_cycles": config.max_cycles})
    env = env_creator_func(render_mode=None, **env_creator_args)
    env.reset()
    
    n_discrete_actions = env.action_space(env.possible_agents[0]).n
    if hasattr(env.observation_space(env.possible_agents[0]), "shape") and len(env.observation_space(env.possible_agents[0]).shape) > 0:
        observation_space = env.observation_space(env.possible_agents[0]).shape[0]
    else:
        observation_space = env.observation_space(env.possible_agents[0]).n
    
    env_name = setup_vars["env_name"]
    device = setup_vars["device"]
    action_wrapper = action_wrappers.get_wrapper(env_name, n_discrete_actions, device)
    sensor_wrapper = sensor_wrappers.get_wrapper(env_name)(env, observation_space, device)
    
    shield_selector = ShieldSelector(
        env_name=env_name, 
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
    
    algo = ALL_ALGORITHMS[config.algo](
        env=env, 
        observation_space=observation_space,
        n_discrete_actions=n_discrete_actions,
        action_wrapper=action_wrapper,
        sensor_wrapper=sensor_wrapper,
        sh_params=sh_params,
        algorithm_params={k: v for k, v in vars(config).items() if k in sh_params},
        alpha=config.shield_alpha
    )
    model_path = f"models/{env_name}/{config.algo}_{setup_vars['cur_time']}/ep{config.max_eps-1}"
    algo.load(model_path)
    
    for ep in range(5):
        reward_hist = eval_episode(env, algo, config.max_cycles)
        print(f"Episode {ep} reward:", {a: np.sum(reward_hist[a]) for a in reward_hist.keys()})
    env.close()

def main():
    # Initialize wandb first so that wandb.config is available.
    # The command-line sweep will override values in wandb.config.
    wandb.init(project=f"{config.wandb_project_prefix}_{os.name}", mode="online" if config.use_wandb else "disabled")
    update_config_from_wandb()
    
    # Now that the config is updated, set up the experiment.
    setup_vars = setup_experiment()
    
    # Run the training loop.
    train_stats = train_loop(setup_vars)
    
    # Save experiment data and plots.
    save_data(setup_vars, train_stats)
    plot_results(setup_vars, train_stats)
    
    # Optionally, run offline evaluation.
    # offline_evaluation(setup_vars)
    
    print("Training complete.")

if __name__ == "__main__":
    main()