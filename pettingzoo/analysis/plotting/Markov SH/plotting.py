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

topics = ["plant_training", "stag_training", "stag_pen_training", "total_reward_mean", "eval_mean_safety", "eval_total_reward_mean"]

percent_rolling = 0.1

groups = get_non_empty_groups()

# baseline for both: IPPO
# Partial shield
group_0_keys = [name(1.0, "IPPO", 0), name(0.5, "SIPPO", 7), name(0.5, "SMAPPO", 7), name(0.5, "SACSPPO", 7)]
group_1_keys = [name(1.0, "IPPO", 0), name(1.0, "SIPPO", 0), name(1.0, "SIPPO", 7)]#, name(0.5, "SACSPPO", 7)]

group_keys = [group_1_keys, group_0_keys]
group_labels = ["full_shield", "partial_shield"]
group_pick = 1

labels_part = ["IPPO", "SIPPO", "SCSPPO", "SPSPPO"]
labels_full = ["IPPO", r"SIPPO ($\mathcal{T}_{weak}$)", r"SIPPO ($\mathcal{T}_{strong}$)"]

labels_all = [labels_full, labels_part]

def sign(x): return (abs(x)/x)
def boundary(x): return x + sign(x)*0.01*x # add 1% to the boundary

# min_vals = {'total_reward_mean': -175.33333333333334, 'eval_mean_safety': -0.00000001, 'eval_total_reward_mean': -137.66666666666666}
# max_vals = {'total_reward_mean': 130.66666666666666, 'eval_mean_safety': 1, 'eval_total_reward_mean': 116.66666666666667}
# min_vals = {'total_reward_mean': -180, 'eval_mean_safety': 0.75, 'eval_total_reward_mean': -140}
# max_vals = {'total_reward_mean': 180, 'eval_mean_safety': 1, 'eval_total_reward_mean': 140}
min_vals = {'plant_training': 0, 'stag_training': 0, 'stag_pen_training': 0}
max_vals = {'plant_training': None, 'stag_training': None, 'stag_pen_training': None}

for topic in topics:
    print()
    print("Topic:", topic)
    for group_pick in range(len(group_keys)):
        plt.clf()
        labels = labels_all[group_pick]
        save_ext = f"_{group_labels[group_pick]}_msh.png"
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

        window = int(len(dfs[0])*percent_rolling)
        fig, ax = plt.subplots()

        # print("Topic:", topic)
        for i, df in enumerate(dfs):
            x = df.rolling(window).mean().index if "eval" not in topic else df.rolling(window).mean().index*10
            if topic == "eval_mean_safety" and i == 0:
                continue
            ax.plot(x, df["mean"].rolling(window).mean(), label=labels[i], color=colors[i])
            ax.fill_between(x, (df["mean"] - df["std"]).rolling(window).mean(), (df["mean"] + df["std"]).rolling(window).mean(), alpha=0.2, color=colors[i])
            # print mean and std of last window
            if labels[i].startswith("IPPO") and "partial" in group_labels[group_pick]:
                continue
            # print(f"{group_labels[group_pick]}: {labels[i]}: {df['mean'].rolling(window).mean().iloc[-1]:.3f} ± {df['std'].rolling(window).mean().iloc[-1]:.3f}")
            print(f"{df['mean'].rolling(window).mean().iloc[-1]:.3f}±{df['std'].rolling(window).mean().iloc[-1]:.3f}")

        ax.set_xlabel("Episode")
        ax.grid()
        # ax.set_ylim(boundary(min_vals[topic]), boundary(max_vals[topic]))
        # ax.set_ylim(min_vals[topic], max_vals[topic])
        # print("Y-Limits:", ax.get_ylim())
        plt.tight_layout()
        

        if topic == "plant_training":
            ax.set_ylabel("Count")
            ax.set_title("Mean Plants Harvested per Episode (Training)")
            ax.legend(loc="upper left")
            plt.savefig(f"plant{save_ext}", dpi=300, bbox_inches="tight")

        if topic == "stag_training":
            ax.set_ylabel("Count")
            ax.set_title("Mean Stags Hunted per Episode (Training)")
            plt.savefig(f"stag{save_ext}", dpi=300, bbox_inches="tight")

        if topic == "stag_pen_training":
            ax.set_ylabel("Count")
            ax.set_title("Mean Stag Penalties per Episode (Training)")
            plt.savefig(f"stag_pen{save_ext}", dpi=300, bbox_inches="tight")

        if topic == "total_reward_mean":
            ax.set_ylabel("Reward")
            ax.set_title("Mean Reward per Episode (Training)")
            plt.savefig(f"training{save_ext}", dpi=300, bbox_inches="tight")

        if topic == "eval_mean_safety":
            ax.set_ylabel("Safety")
            ax.set_title("Mean Safety per Episode (Evaluation)")
            plt.savefig(f"safety{save_ext}", dpi=300, bbox_inches="tight")

        if topic == "eval_total_reward_mean":
            ax.set_ylabel("Reward")
            ax.set_title("Mean Reward per Episode (Evaluation)")
            plt.savefig(f"eval{save_ext}", dpi=300, bbox_inches="tight")

"""
\begin{table*}[]
\centering
\begin{tabular}{c|cccccc}
\hline
\multicolumn{1}{c|}{\textbf{Algorithm}} & \multicolumn{1}{c}{$\sum$\textbf{plants}} & \multicolumn{1}{c}{$\sum$\textbf{stags}} & \multicolumn{1}{c}{$\sum$\textbf{penalties}} & \multicolumn{1}{c}{\textbf{$\sum r_{\text{train}}$)}} & \multicolumn{1}{c}{\textbf{$\sum r_{\text{eval}}$}} & \multicolumn{1}{c}{\textbf{mean safety}} \\ \hline
IPPO                                                  & 18.740±4.984                                  & 0.093±0.113                               & 5.500±2.036                                  & 13.707±4.969                                     & 16.067±5.803                                 & -                                      \\
SIPPO ($\mathcal{T}_{weak}$)                          & 18.733±4.257                                  & 12.800±7.405                              & 14.253±3.566                                 & 68.480±35.118                                    & 79.800±38.332                                & 0.955±0.014                            \\
SIPPO ($\mathcal{T}_{weak}$)                          & 11.333±4.076                                  & 77.320±8.815                              & 11.073±3.249                                 & 386.860±46.507                                   & 393.067±65.274                               & 0.954±0.010                            \\ \hline
\end{tabular}
\end{table*}

\begin{table*}[]
\centering
\begin{tabular}{c|cccccc}
\hline
\multicolumn{1}{c|}{\textbf{Algorithm}} & \multicolumn{1}{c}{$\sum$\textbf{plants}} & \multicolumn{1}{c}{$\sum$\textbf{stags}} & \multicolumn{1}{c}{$\sum$\textbf{penalties}} & \multicolumn{1}{c}{\textbf{$\sum r_{\text{train}}$)}} & \multicolumn{1}{c}{\textbf{$\sum r_{\text{eval}}$}} & \multicolumn{1}{c}{\textbf{mean safety}} \\ \hline
IPPO                                    & 18.740±4.984                                  & 0.093±0.113                               & 5.500±2.036                                  & 13.707±4.969                                     & 16.067±5.803                                 & -                                                    \\
SIPPO                                   & 10.540±3.727                                  & 3.093±2.067                               & 8.953±3.865                                  & 17.053±10.989                                    & 10.867±10.108                                & 0.580±0.051                                          \\
SCSPPO                                  & 10.773±3.864                                  & 3.160±2.185                               & 8.760±3.319                                  & 17.813±11.485                                    & 13.333±6.030                                 & 0.613±0.035                                          \\
SPSPPO                                  & 8.153±3.311                                   & 30.760±12.496                             & 18.300±4.184                                 & 143.653±64.459                                   & 145.667±73.443                               & 0.722±0.076                                          \\ \hline
\end{tabular}
\end{table*}

"""