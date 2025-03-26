from process_api import get_non_empty_groups, name
import pandas as pd
import matplotlib.pyplot as plt
import os

# x-axis = shielded ratio

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

# get default colour wheel for matplotlib
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# ingore warnings
import warnings
warnings.filterwarnings("ignore")

topics = ["eval_mean_reward", "eval_mean_safety"]
percent_rolling = 0.1

groups = get_non_empty_groups()

# groups are on the basis of bars that will be plotted side by side

mus = ["[0.5, 1]", "[1, 1]", "[1.5, 1]", "[2.5, 1]", "[5, 1]"]

files = os.listdir()
files = [x for x in files if x.endswith(".csv")]

for mu in mus:
    group_keys = []

    n_shielded = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    algos = ["SIPPO", "SMAPPO", "SACSPPO"]
    for shielded_ratio in n_shielded:
        group_keys.append([])
        for algo in algos:
            group_keys[-1].append(name(algo, mu, shielded_ratio))

    group_labels = n_shielded
    labels = algos

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
            # print(df.columns)

            cols = [x for x in df.columns if x.endswith(topic)]
            df = df[cols]

            dfs = []
            for item in plot_group:
                try:
                    dfs.append(df[[x+" - "+topic for x in item]])
                except KeyError:
                    print(f"KeyError: {item}")
                    continue
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
        width = 0.25
        x = range(len(group_labels))
        ax.bar(x, [means[group_label]["SIPPO"] for group_label in group_labels], width, label="SIPPO", yerr=[stds[group_label]["SIPPO"] for group_label in group_labels], capsize=5)
        ax.bar([i+width for i in x], [means[group_label]["SMAPPO"] for group_label in group_labels], width, label="SCSPPO", yerr=[stds[group_label]["SMAPPO"] for group_label in group_labels], capsize=5)
        ax.bar([i+2*width for i in x], [means[group_label]["SACSPPO"] for group_label in group_labels], width, label="SACSPPO", yerr=[stds[group_label]["SACSPPO"] for group_label in group_labels], capsize=5)
        print(f"SIPPO: ", [round(means[group_label]["SIPPO"],3) for group_label in group_labels], [round(stds[group_label]["SIPPO"],3) for group_label in group_labels])
        print(f"SCSPPO: ", [round(means[group_label]["SMAPPO"],3) for group_label in group_labels], [round(stds[group_label]["SMAPPO"],3) for group_label in group_labels])
        print(f"SACSPPO: ", [round(means[group_label]["SACSPPO"],3) for group_label in group_labels], [round(stds[group_label]["SACSPPO"],3) for group_label in group_labels])

        ax.set_xlabel("Shielded Ratio")
        ax.set_xticks([i+width for i in x])
        ax.set_xticklabels(group_labels)       

        if topic == "mean_reward":
            ax.set_ylabel("Reward")
            ax.legend()
            ax.set_title("Mean Reward per Episode (Training)")
            save_ext = "_training_ma_epgg.png"

        if topic == "eval_mean_reward":
            ax.set_ylabel("Reward")
            ax.set_title("Mean Reward per Episode (Evaluation)")
            save_ext = "_evaluation_ma_epgg.png"

        if topic == "eval_mean_safety":
            ax.set_ylabel("Action==Cooperate")
            ax.set_ylim(-0.05,1.05)
            ax.set_title("Mean Cooperation per Episode (Evaluation)")
            save_ext = "_safety_ma_epgg.png"

        mu_ext = mu.replace('.','_').replace('[', '').replace(']', '').replace(" ", "").replace(",", "_")
        print(f"Saving to: images/bar_{mu_ext}_{save_ext}")
        plt.savefig(f"images/bar_{mu_ext}_{save_ext}", dpi=300, bbox_inches="tight")
        # plt.savefig(f"bar_{save_ext}", dpi=300, bbox_inches="tight")
        # exit()