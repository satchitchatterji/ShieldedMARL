import os
from env_selection import ALL_ENVS

class ShieldSelector:
    def __init__(self, env_name, n_actions=None, n_sensors=None, filename = "", version=0):
        
        self.base_dir = "shields"
        self.env_name = env_name
        self.n_actions = n_actions
        self.n_sensors = n_sensors

        assert filename or (n_actions and n_sensors), "Either filename or (n_actions and n_sensors) must be provided."

        if filename != "" and filename != "default":
            print("Using provided filename as shield program.")
            self.file = filename

        supported_envs = ALL_ENVS.keys()
        assert env_name in supported_envs, f"Shield program for {env_name} not supported."
        if filename == "default":
            self.file = f"{env_name}/shield_v{version}.pl"

        self.verify()

    def verify(self):
        full_path = os.path.join(self.base_dir, self.file)
        assert os.path.exists(full_path), f"Shield program not found at {full_path}."
        print(f"[INFO] Shield program found at {full_path}.")
