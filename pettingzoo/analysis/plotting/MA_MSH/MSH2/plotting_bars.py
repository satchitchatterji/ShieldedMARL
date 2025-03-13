from process_api import get_non_empty_groups, name
import pandas as pd
import matplotlib.pyplot as plt
import os

# x-axis = n_agents
# bars = algo

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=11)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# get default colour wheel for matplotlib
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# ingore warnings
import warnings
warnings.filterwarnings("ignore")

topics = ["plant_training", "stag_training", "stag_pen_training", "total_reward_mean", "eval_mean_safety", "eval_total_reward_mean"]
percent_rolling = 0.1

groups = get_non_empty_groups()

group_keys = []
group_labels = ["2", "3", "4", "5"]
labels = ["IPPO", "MAPPO", "ACSPPO", "SIPPO", "SMAPPO", "SACSPPO"]
for n_agents in group_labels:
    group_keys.append([])
    for algo in labels:
        group_keys[-1].append(name(n_agents, algo))

files = os.listdir()
files = [x for x in files if x.endswith(".csv")]

for topic in topics:
    print("Topic:", topic)
    means = {group_label:{label:0 for label in labels} for group_label in group_labels}
    stds = {group_label:{label:0 for label in labels} for group_label in group_labels}

    for group_pick in range(len(group_keys)):
        plt.clf()
        
        plot_group = [groups[x] for x in group_keys[group_pick]]

        # find file with topic in one of the columns
        correct_file = ""
        for file in files:
            with open(file, "r") as f:
                line = f.readline()
                if "- "+topic in line:
                    correct_file = file
                    break
        
        # print("File:", file)
        df = pd.read_csv(correct_file)
        # restrict df to first 500 episodes
        df = df.iloc[:500]
        if topic == "eval_mean_safety":
            df = df.iloc[:20]

        # print(df.columns)

        cols = [x for x in df.columns if x.endswith(topic)]
        df = df[cols]

        dfs = []
        for item in plot_group:
            dfs.append(df[[x+" - "+topic for x in item]])
            # find last row with all non-nan values and remove all rows after it
            for i in range(len(dfs[-1])-1, -1, -1):
                if not dfs[-1].iloc[i].isna().any():
                    dfs[-1] = dfs[-1].iloc[:i+1]
                    break
            dfs[-1]["mean"] = dfs[-1].mean(axis=1)
            dfs[-1]["std"] = dfs[-1].std(axis=1)
            # print(dfs[-1].iloc[-1])
            # exit()

        # collect the last rolling means and stds
        for i in range(len(dfs)):
            means[group_labels[group_pick]][labels[i]] = dfs[i]["mean"].rolling(int(percent_rolling*len(dfs[i]["mean"]))).mean().iloc[-1]
            stds[group_labels[group_pick]][labels[i]] = dfs[i]["std"].rolling(int(percent_rolling*len(dfs[i]["std"]))).mean().iloc[-1]
    
    fig, ax = plt.subplots()
    ax.grid()
    plt.tight_layout()

    n_bars = len(group_labels)
    width = 0.125
    x = range(len(group_labels))
    ax.bar(x, [means[group_label]["IPPO"] for group_label in group_labels], width, label="IPPO", yerr=[stds[group_label]["IPPO"] for group_label in group_labels], capsize=5)
    ax.bar([i+width for i in x], [means[group_label]["MAPPO"] for group_label in group_labels], width, label="CSPPO", yerr=[stds[group_label]["MAPPO"] for group_label in group_labels], capsize=5)
    ax.bar([i+2*width for i in x], [means[group_label]["ACSPPO"] for group_label in group_labels], width, label="ACSPPO", yerr=[stds[group_label]["ACSPPO"] for group_label in group_labels], capsize=5)
    ax.bar([i+3*width for i in x], [means[group_label]["SIPPO"] for group_label in group_labels], width, label="SIPPO", yerr=[stds[group_label]["SIPPO"] for group_label in group_labels], capsize=5)
    ax.bar([i+4*width for i in x], [means[group_label]["SMAPPO"] for group_label in group_labels], width, label="SCSPPO", yerr=[stds[group_label]["SMAPPO"] for group_label in group_labels], capsize=5)
    ax.bar([i+5*width for i in x], [means[group_label]["SACSPPO"] for group_label in group_labels], width, label="SACSPPO", yerr=[stds[group_label]["SACSPPO"] for group_label in group_labels], capsize=5)
    # ax.bar(x, [means[group_label]["IPPO"] for group_label in group_labels], width, label="IPPO", yerr=[stds[group_label]["IPPO"] for group_label in group_labels], capsize=5)
    # ax.bar([i+width for i in x], [means[group_label]["SIPPO"] for group_label in group_labels], width, label="SIPPO", yerr=[stds[group_label]["SIPPO"] for group_label in group_labels], capsize=5)
    # print(f"IPPO: ", [round(means[group_label]["IPPO"],3) for group_label in group_labels], [round(stds[group_label]["IPPO"],3) for group_label in group_labels])
    # print(f"SIPPO: ", [round(means[group_label]["SIPPO"],3) for group_label in group_labels], [round(stds[group_label]["SIPPO"],3) for group_label in group_labels])
    
    ax.set_xlabel("Number of Agents")
    ax.set_xticks([i+2.5*width for i in x])
    ax.set_xticklabels(group_labels)       

    # if topic == "mean_reward":
    #     ax.set_ylabel("Reward")
    #     ax.legend()
    #     ax.set_title("Mean Reward per Episode (Training)")
    #     save_ext = "_training_msh2.png"

    # topics = ["plant_training", "stag_training", "stag_pen_training", "total_reward_mean", "eval_mean_safety", "eval_total_reward_mean"]

    if topic == "plant_training":
        ax.set_ylabel("Plants Harvested per Episode")
        ax.legend()
        ax.set_title("Plants Harvested (Training)")
        save_ext = "_plant_training_msh2.png"

    elif topic == "stag_training":
        ax.set_ylabel("Stags Hunted per Episode")
        ax.legend()
        ax.set_title("Stags Hunted (Training)")
        save_ext = "_stag_training_msh2.png"

    elif topic == "stag_pen_training":
        ax.set_ylabel("Stag Penalties Attained per Episode")
        ax.legend()
        ax.set_title("Stag Penalties (Training)")
        save_ext = "_stag_pen_training_msh2.png"

    elif topic == "total_reward_mean":
        ax.set_ylabel("Total Reward per Episode")
        ax.legend()
        ax.set_title("Total Reward (Training)")
        save_ext = "_total_reward_mean_msh2.png"

    elif topic == "eval_mean_safety":
        ax.set_ylabel("Mean Safety Score")
        ax.legend()
        ax.set_title("Mean Safety Score (Evaluation)")
        save_ext = "_eval_mean_safety_msh2.png"

    elif topic == "eval_total_reward_mean":
        ax.set_ylabel("Total Reward per Episode")
        ax.legend()
        ax.set_title("Total Reward (Evaluation)")
        save_ext = "_eval_total_reward_mean_msh2.png"


    plt.savefig(f"bar{save_ext}", dpi=300, bbox_inches="tight")
