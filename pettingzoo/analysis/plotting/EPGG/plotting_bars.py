from process_api import get_non_empty_groups, name
import pandas as pd
import matplotlib.pyplot as plt
import os

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# ingore warnings
import warnings
warnings.filterwarnings("ignore")

topics = ["mean_reward", "eval_mean_reward", "eval_mean_safety"]

percent_rolling = 0.1

groups = get_non_empty_groups()

mu = str([0.5,1])
group_0_keys = [name("IPPO", mu, 1), name("SIPPO", mu, 3)]
mu = str([1,1])
group_1_keys = [name("IPPO", mu, 1), name("SIPPO", mu, 3)]
mu = str([1.5,1])
group_2_keys = [name("IPPO", mu, 1), name("SIPPO", mu, 3)]
mu = str([2.5,1])
group_3_keys = [name("IPPO", mu, 1), name("SIPPO", mu, 3)]
mu = str([5,1])
group_4_keys = [name("IPPO", mu, 3), name("SIPPO", mu, 3)]


group_keys = [group_0_keys, group_1_keys, group_2_keys, group_3_keys, group_4_keys]
# group_labels = ["mu0_5", "mu1", "mu1_5", "mu2_5", "mu5"]
group_labels = ["0.5", "1", "1.5", "2.5", "5"]

labels = ["IPPO", "SIPPO"]


for topic in topics:
    print("Topic:", topic)
    means = {group_label:{"IPPO":0, "SIPPO":0} for group_label in group_labels}
    stds = {group_label:{"IPPO":0, "SIPPO":0} for group_label in group_labels}
    for group_pick in range(len(group_keys)):
        plt.clf()
        
        plot_group = [groups[x] for x in group_keys[group_pick]]

        # print(plot_group)
        # exit()

        # get all csv files in directory
        files = os.listdir()
        files = [x for x in files if x.endswith(".csv")]

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
        # print(df.columns)

        cols = [x for x in df.columns if x.endswith(topic)]
        df = df[cols]

        # get default colour wheel for matplotlib
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        dfs = []

        for item in plot_group:
            dfs.append(df[[x+" - "+topic for x in item]])
            dfs[-1]["mean"] = dfs[-1].mean(axis=1)
            dfs[-1]["std"] = dfs[-1].std(axis=1)   


        # collect the last 10 means and stds

        for i, df in enumerate(dfs):
            # x = df.rolling(window).mean().index if topic == "mean_reward" else df.rolling(window).mean().index*10
            # means.append(df["mean"].iloc[-10:].mean()]
            # stds.append([df["std"].iloc[-10:].mean()
            
            means[group_labels[group_pick]][labels[i]] = df["mean"].iloc[-10:].mean()
            stds[group_labels[group_pick]][labels[i]] = df["std"].iloc[-10:].mean()
    
    fig, ax = plt.subplots()
    ax.grid()
    plt.tight_layout()
    width = 0.35
    x = range(len(group_labels))
    ax.bar(x, [means[group_label]["IPPO"] for group_label in group_labels], width, label="IPPO", yerr=[stds[group_label]["IPPO"] for group_label in group_labels], capsize=5)
    ax.bar([i+width for i in x], [means[group_label]["SIPPO"] for group_label in group_labels], width, label="SIPPO", yerr=[stds[group_label]["SIPPO"] for group_label in group_labels], capsize=5)
    print(f"IPPO: ", [round(means[group_label]["IPPO"],3) for group_label in group_labels], [round(stds[group_label]["IPPO"],3) for group_label in group_labels])
    print(f"SIPPO: ", [round(means[group_label]["SIPPO"],3) for group_label in group_labels], [round(stds[group_label]["SIPPO"],3) for group_label in group_labels])
    
    ax.set_xlabel("$\mu$")
    ax.set_xticks([i+width/2 for i in x])
    ax.set_xticklabels(group_labels)       

    if topic == "mean_reward":
        ax.set_ylabel("Reward")
        ax.legend()
        ax.set_title("Mean Reward per Episode (Training)")
        save_ext = "_training_epgg.png"

    if topic == "eval_mean_reward":
        ax.set_ylabel("Reward")
        ax.set_title("Mean Reward per Episode (Evaluation)")
        save_ext = "_evaluation_epgg.png"

    if topic == "eval_mean_safety":
        ax.set_ylabel("Action==Cooperate")
        ax.set_ylim(-0.05,1.05)
        ax.set_title("Mean Cooperation per Episode (Evaluation)")
        save_ext = "_safety_epgg.png"

    plt.savefig(f"bar_safety{save_ext}", dpi=300, bbox_inches="tight")


"""
\begin{table}[]
\centering
\begin{tabular}{l|lll}
\hline
\multicolumn{1}{c|}{\textbf{Algorithm}} & \multicolumn{1}{c}{\textbf{$\bar{r}$ (training)}} & \multicolumn{1}{c}{\textbf{$\bar{r}$ (eval)}} & \multicolumn{1}{c}{\textbf{safety/cooperation}} \\ \hline
IPPO                                    & 1.985 ± 0.025                                     & 1.990 ± 0.019                                 & 0.006 ± 0.005                                   \\
SIPPO ($\mathcal{T}_{pure}$)            & 4.000 ± 0.000                                     & 4.000 ± 0.000                                 & 1.000 ± 0.000                                   \\
SIPPO ($\mathcal{T}_{mixed}$)           & 2.574 ± 0.477                                     & 2.633 ± 0.424                                 & 0.576 ± 0.081                                   \\ \hline
\end{tabular}
\end{table}

"""