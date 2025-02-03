import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# df = pd.read_csv('wandb_export_2024-07-01T14_20_08.782+02_00.csv'); topic = "mean_reward"
# df = pd.read_csv('wandb_export_2024-07-01T14_19_47.246+02_00.csv'); topic = "eval_mean_safety"
# df = pd.read_csv('wandb_export_2024-07-01T15_15_17.076+02_00.csv'); topic = "eval_mean_reward"

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

topics = ["mean_reward", "eval_mean_reward", "eval_mean_safety"]

for topic in topics:
    print("Topic:", topic)
    df = pd.read_csv(f'{topic}.csv')

    percent_rolling = 0.1

    cols = [x for x in df.columns if x.endswith(topic)]
    df = df[cols]
    firstunderscore = [x.find("_") for x in cols] 
    firstspace = [x.find(" ") for x in cols] 

    algos = ["IPPO", "SIPPO"]

    # get default colour wheel for matplotlib
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # red, light blue, blue, dark blue
    # colors = [colors[1], "darkblue", "lightblue", "blue"]

    ippo_cols = [cols[i] for i in range(len(cols)) if cols[i][:firstunderscore[i]] == "IPPO"]
    sippo_cols = [cols[i] for i in range(len(cols)) if cols[i][:firstunderscore[i]] == "SIPPO"]


    ippo_df = df[ippo_cols]
    sippo_df = df[[x for x in sippo_cols]]

    # take mean and std of each algo
    ippo_df["mean"] = ippo_df.mean(axis=1)
    sippo_df["mean"] = sippo_df.mean(axis=1)

    ippo_df["std"] = ippo_df.std(axis=1)
    sippo_df["std"] = sippo_df.std(axis=1)

    window = int(len(ippo_df)*percent_rolling)
    # print("Rolling Window:", window)

    # plot
    fig, ax = plt.subplots()
    x = ippo_df.rolling(window).mean().index if topic == "mean_reward" else ippo_df.rolling(window).mean().index*10
    ax.plot(x, ippo_df["mean"].rolling(window).mean(), label="IPPO", color=colors[0])
    ax.fill_between(x, (ippo_df["mean"] - ippo_df["std"]).rolling(window).mean(), (ippo_df["mean"] + ippo_df["std"]).rolling(window).mean(), alpha=0.2, color=colors[0])

    x = sippo_df.rolling(window).mean().index if topic == "mean_reward" else sippo_df.rolling(window).mean().index*10
    ax.plot(x, sippo_df["mean"].rolling(window).mean(), label="SIPPO", color=colors[1])
    ax.fill_between(x, (sippo_df["mean"] - sippo_df["std"]).rolling(window).mean(), (sippo_df["mean"] + sippo_df["std"]).rolling(window).mean(), alpha=0.2, color=colors[1])

    print(f"IPPO: {ippo_df['mean'].rolling(window).mean().iloc[-1]:.3f} ± {ippo_df['std'].rolling(window).mean().iloc[-1]:.3f}")
    print(f"SIPPO: {sippo_df['mean'].rolling(window).mean().iloc[-1]:.3f} ± {sippo_df['std'].rolling(window).mean().iloc[-1]:.3f}")

    ax.set_xlabel("Episode")
    ax.grid()
    plt.tight_layout()

    if topic == "mean_reward":
        ax.set_ylabel("Mean Reward")
        ax.set_title("Mean Reward per Episode Step (Training)")
        ax.legend()
        plt.savefig("training_pd.png", dpi=300, bbox_inches="tight")

    if topic == "eval_mean_safety":
        ax.set_ylabel("Mean Action==Cooperation")
        ax.set_title("Mean Cooperation (Safety) per Episode")
        plt.savefig("safety_pd.png", dpi=300, bbox_inches="tight")

    if topic == "eval_mean_reward":
        ax.set_ylabel("Mean Reward")
        ax.set_title("Mean Reward per Episode Step (Evaluation)")
        plt.savefig("eval_pd.png", dpi=300, bbox_inches="tight")


    # plt.show()