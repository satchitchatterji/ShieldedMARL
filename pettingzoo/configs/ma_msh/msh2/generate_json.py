import json

nseeds = 5

with open('template.json') as f:
    base_config = json.load(f)

algos = ["IPPO", "MAPPO", "ACSPPO", "SIPPO", "SMAPPO", "SACSPPO"]
nagents = [2,3,4,5]


filenames = []
for algo in algos:
    for na in nagents:
        base_config["algo"] = algo
        base_config["env_config"]["n_agents"] = na
        filenames.append(f"{algo}_{str(na).replace('.','')}.json")
        with open(filenames[-1], "w") as f:
            s = json.dumps(base_config, indent=4)
            f.write(s)

runfile = f"""#!/bin/bash

#SBATCH --partition=rome
#SBATCH --job-name=Debug_MultiAgent
#SBATCH --ntasks={len(filenames)*nseeds}
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --output=/home/schatterji1/Debug_MultiAgent/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate pls

cd $HOME/ShieldedMARL/pettingzoo

""" +\
" &\n".join([f"srun -n {nseeds} python main.py --config=configs/ma_msh/msh2/{filename}" for filename in filenames])

with open("run.job", "w") as f:
    f.write(runfile)
