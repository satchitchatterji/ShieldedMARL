import pandas as pd 
import wandb
import tqdm
api = wandb.Api()

# Project is specified by <entity/project-name>
project = "satchat/snellius_mamsh2_posix_markov_stag_hunt"
keys = ["plant_training", "stag_training", "stag_pen_training", "total_reward_mean", "eval_mean_safety", "eval_total_reward_mean"]

redownload = True

runs = api.runs(project)

summary_list, config_list, name_list, path_list = [], [], [], []
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
    path_list.append(run.path[-1])

runs_df = pd.DataFrame({
    "config": config_list,
    "name": name_list,
    "path": path_list,
    })


runs_df.to_csv("project.csv")

# exit()

dfs = {key: list() for key in keys}
for run in tqdm.tqdm(runs, desc="Downloading runs"):
    name = run.name
    path = run.path[-1]
    for key in keys:
        col_name = f"{name} - {path} - {key}"
        if not redownload:
            existing_df = pd.read_csv(f"{key}.csv")
            if col_name in existing_df.columns:
                dfs[key].append(existing_df[col_name])
            else:
                try:
                    hist_df = run.history(keys=[key], pandas=True)
                    hist_df = hist_df.rename(columns={key: col_name})
                    dfs[key].append(hist_df[col_name])
                except KeyError as e:
                    print(f"Key Error: {e}")

        else:
            try:
                hist_df = run.history(keys=[key], pandas=True)
                hist_df = hist_df.rename(columns={key: col_name})
                dfs[key].append(hist_df[col_name])
            except KeyError as e:
                print(f"Key Error: {e}")

for key in keys:
    df = pd.concat(dfs[key], axis=1)
    df.to_csv(f"{key}.csv")