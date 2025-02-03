import pandas as pd
import numpy as np

def get_new_seed(config):
    if config.seed == -1:
        run_histories = pd.read_csv("histories/configs/run_history.csv")
        config.seed = int(run_histories["seed"].dropna().max()+1)
        print(f"[INFO] Seed not provided. Using new seed: {config.seed}")

    return config.seed