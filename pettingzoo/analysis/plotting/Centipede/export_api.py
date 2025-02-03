import pandas as pd 
import wandb
import tqdm
api = wandb.Api()

sippo_cols = ["SIPPO_2024-07-05_225742", "SIPPO_2024-07-05_225708", "SIPPO_2024-07-05_225444", "SIPPO_2024-07-05_225439", "SIPPO_2024-07-05_225412"]
ippo_cols = ["IPPO_2024-07-05_224517", "IPPO_2024-07-05_224514", "IPPO_2024-07-05_224318"]

explore_policy_softmax_SIQL = ["SIQL_2024-07-05_232934", "SIQL_2024-07-05_232850", "SIQL_2024-07-05_232837"]
explore_policy_e_greedy_SIQL = ["SIQL_2024-07-05_231826", "SIQL_2024-07-05_231821", "SIQL_2024-07-05_231749"]
explore_policy_softmax_IQL = ["IQL_2024-07-05_232241", "IQL_2024-07-05_232219", "IQL_2024-07-05_232142"]
explore_policy_e_greedy_IQL = ["IQL_2024-07-05_231433", "IQL_2024-07-05_231427", "IQL_2024-07-05_231424"]

all_runs = ippo_cols + sippo_cols + explore_policy_softmax_SIQL + explore_policy_e_greedy_SIQL + explore_policy_softmax_IQL + explore_policy_e_greedy_IQL

# Project is specified by <entity/project-name>
project = "satchat/nt_centipede"

keys = ["total_reward_mean", "eval_total_reward_mean", "eval_mean_safety"]

redownload = True

runs = api.runs(project)

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "config": config_list,
    "name": name_list
    })


runs_df.to_csv("project.csv")

dfs = {key: list() for key in keys}
for run in tqdm.tqdm(runs, desc="Downloading runs"):
    name = run.name
    if name not in all_runs:
        continue
    for key in keys:
        col_name = f"{name} - {key}"
        
        if not redownload:
            existing_df = pd.read_csv(f"{key}.csv")
            if col_name in existing_df.columns:
                dfs[key].append(existing_df[col_name])
            else:
                try:
                    hist_df = run.history(keys=[key], pandas=True)
                    hist_df = hist_df.rename(columns={key: f"{name} - {key}"})
                    dfs[key].append(hist_df[col_name])
                except KeyError as e:
                    print(f"Key Error: {e}")

        else:
            try:
                hist_df = run.history(keys=[key], pandas=True)
                hist_df = hist_df.rename(columns={key: f"{name} - {key}"})
                dfs[key].append(hist_df[col_name])
            except KeyError as e:
                print(f"Key Error: {e}")

for key in keys:
    df = pd.concat(dfs[key], axis=1)
    df.to_csv(f"{key}.csv")