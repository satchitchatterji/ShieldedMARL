import pandas as pd
import pprint


# get unique values for each field
def name(r,a):
    return f"{r}_{a}"

def get_non_empty_groups():

    df = pd.read_csv("project.csv")
    # make df from config column
    config_df = pd.DataFrame.from_dict([eval(x) for x in df["config"].to_list()])
    # print(config_df)

    config_df["name"] = df["name"]
    config_df["path"] = df["path"]

    group_fields = ["shielded_ratio", "algo"]

    runs = {}
    # Iterate through each unique combination of field values
    for r in config_df["shielded_ratio"].unique():
        for a in config_df["algo"].unique():
            filtered_df = config_df[
                (config_df["shielded_ratio"] == r) &
                (config_df["algo"] == a)
            ]
            # filtered_df = filtered_df[~filtered_df["name"].isin(exclude_runs)]
            # Store the filtered DataFrame in the dictionary
            runs[name(r,a)] = [n+" - "+p for n,p in zip(filtered_df["name"].tolist() , filtered_df["path"].tolist())]

    # delete empty groups
    runs = {k:v for k,v in runs.items() if v}
    pprint.pp(runs)
    return runs

if __name__ == "__main__":
    print(get_non_empty_groups())
