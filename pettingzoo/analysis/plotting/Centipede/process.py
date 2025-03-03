import pandas as pd
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore")

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

percent_rolling = 0.1

sippo_cols = ["SIPPO_2024-07-05_225742", "SIPPO_2024-07-05_225708", "SIPPO_2024-07-05_225444", "SIPPO_2024-07-05_225439", "SIPPO_2024-07-05_225412"]
ippo_cols = ["IPPO_2024-07-05_224517", "IPPO_2024-07-05_224514", "IPPO_2024-07-05_224318"]

explore_policy_softmax_SIQL = ["SIQL_2024-07-05_232934", "SIQL_2024-07-05_232850", "SIQL_2024-07-05_232837"]
explore_policy_e_greedy_SIQL = ["SIQL_2024-07-05_231826", "SIQL_2024-07-05_231821", "SIQL_2024-07-05_231749"]
explore_policy_softmax_IQL = ["IQL_2024-07-05_232241", "IQL_2024-07-05_232219", "IQL_2024-07-05_232142"]
explore_policy_e_greedy_IQL = ["IQL_2024-07-05_231433", "IQL_2024-07-05_231427", "IQL_2024-07-05_231424"]

group_ppo = [ippo_cols, sippo_cols]
group_softmax = [explore_policy_softmax_IQL, explore_policy_softmax_SIQL]
group_e_greedy = [explore_policy_e_greedy_IQL, explore_policy_e_greedy_SIQL]


# # topic = "total_reward_mean"
# # topic = "eval_mean_safety"
# topic = "eval_total_reward_mean"

topics = ["total_reward_mean", "eval_mean_safety", "eval_total_reward_mean"]
groups = [group_ppo, group_softmax, group_e_greedy]
all_labels = [["IPPO", "SIPPO"], ["IQL (softmax)", "SIQL (softmax)"], ["IQL ($\epsilon$-greedy)", "SIQL ($\epsilon$-greedy)"]]
save_exts = ["_cent_ppo.png", "_cent_softmax.png", "_cent_e_greedy.png"]

for topic in topics:
    print("\nTopic:", topic)
    for i, group in enumerate(groups):
        labels = all_labels[i]
        save_ext = save_exts[i]
        plot_group = group

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

        # ippo_df = df[[x+" - "+topic for x in ippo_cols]]
        # ippo_df["mean"] = ippo_df.mean(axis=1)
        # ippo_df["std"] = ippo_df.std(axis=1)
        for item in plot_group:
            dfs.append(df[[x+" - "+topic for x in item]])
            # take mean and std of each algo
            dfs[-1]["mean"] = dfs[-1].mean(axis=1)
            dfs[-1]["std"] = dfs[-1].std(axis=1)    

        window = int(len(dfs[0])*percent_rolling)
        # plot
        fig, ax = plt.subplots()

        for i, df in enumerate(dfs):
            x = df.rolling(window).mean().index if topic == "total_reward_mean" else df.rolling(window).mean().index*10
            ax.plot(x, df["mean"].rolling(window).mean(), label=labels[i], color=colors[i])
            ax.fill_between(x, (df["mean"] - df["std"]).rolling(window).mean(), (df["mean"] + df["std"]).rolling(window).mean(), alpha=0.2, color=colors[i])
            # print mean and std of last window
            # print(f"{labels[i]}: {df['mean'].rolling(window).mean().iloc[-1]:.3f} ± {df['std'].rolling(window).mean().iloc[-1]:.3f}")
            print(f"{df['mean'].rolling(window).mean().iloc[-1]:.3f}±{df['std'].rolling(window).mean().iloc[-1]:.3f}")

        ax.set_xlabel("Episode")
        ax.grid()
        plt.tight_layout()

        if topic == "total_reward_mean":
            ax.set_ylabel("Reward")
            ax.set_title("Mean Reward per Episode (Training)")
            ax.legend(loc="center left")
            plt.savefig(f"training{save_ext}", dpi=300, bbox_inches="tight")

        if topic == "eval_mean_safety":
            ax.set_ylabel("Mean Action==Continue")
            ax.set_title("Mean Continue (Safety) per Episode")
            plt.savefig(f"safety{save_ext}", dpi=300, bbox_inches="tight")

        if topic == "eval_total_reward_mean":
            ax.set_ylabel("Reward")
            ax.set_title("Mean Reward per Episode (Evaluation)")
            plt.savefig(f"eval{save_ext}", dpi=300, bbox_inches="tight")


        # plt.show()


"""
\begin{table*}[]
\centering
\begin{tabular}{l|lll}
\hline
\multicolumn{1}{c|}{\textbf{Algorithm}} & \multicolumn{1}{c}{\textbf{$\sum r$ (training)}} & \multicolumn{1}{c}{\textbf{$\sum r$ (eval)}} & \multicolumn{1}{c}{\textbf{mean safety/cooperation}} \\ \hline
IPPO                                    & 42.353±36.618                                    & 42.833±35.812                                & 0.826±0.229                                          \\
SIPPO                                   & 100.500±0.000                                    & 100.500±0.000                                & 1.000±0.000                                          \\ \hline
IQL ($\epsilon$-greedy)                 & 34.620±46.590                                    & 34.700±46.531                                & 0.682±0.228                                          \\
SIQL ($\epsilon$-greedy)                & 100.500±0.000                                    & 100.500±0.000                                & 1.000±0.000                                          \\ \hline
IQL (softmax)                           & 1.727±1.012                                      & 30.100±38.611                                & 0.733±0.210                                          \\
SIQL (softmax)                          & 100.500±0.000                                    & 100.500±0.000                                & 1.000±0.000                                          \\ \hline
\end{tabular}
\end{table*}

"""