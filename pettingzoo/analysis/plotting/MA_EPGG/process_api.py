import pandas as pd
import pprint


runs_too_short = []
killed_runs = []

exclude_runs = runs_too_short + killed_runs

# get unique values for each field
def name(a,f,s):
    return f"{a}_{f.replace('.','_')}_{str(s).replace('.','_')}"

def get_f(config):
    return str(eval(str(config))["f_params"])

def get_non_empty_groups():

    df = pd.read_csv("project.csv")
    # make df from config column
    config_df = pd.DataFrame.from_dict([eval(x) for x in df["config"].to_list()])
    # print(config_df)

    config_df["name"] = df["name"]
    config_df["f_params"] = config_df["env_config"].apply(get_f)
    config_df["path"] = df["path"]

    group_fields = ["algo", "f_params", "shielded_ratio"]

    runs = {}
    # Iterate through each unique combination of field values
    for a in config_df["algo"].unique():
        for f in config_df["f_params"].unique():
            for s in config_df["shielded_ratio"].unique():
                filtered_df = config_df[
                    (config_df["algo"] == a) &
                    (config_df["f_params"] == f) &
                    (config_df["shielded_ratio"] == s)
                ]
                filtered_df = filtered_df[~filtered_df["name"].isin(exclude_runs)]
                # Store the filtered DataFrame in the dictionary
                runs[name(a,f,s)] = [n+" - "+p for n,p in zip(filtered_df["name"].tolist() , filtered_df["path"].tolist())]

    # delete empty groups
    runs = {k:v for k,v in runs.items() if v}
    pprint.pp(runs)
    return runs

if __name__ == "__main__":
    print(get_non_empty_groups())
