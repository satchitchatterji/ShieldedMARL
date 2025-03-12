import json

nseeds = 5

with open('template.json') as f:
    base_config = json.load(f)

algos = ["SIPPO", "SMAPPO", "SACSPPO"]
percs = [round(n/5,3) for n in range(6)]

filenames = []
for algo in algos:
    for perc in percs:
        base_config["algo"] = algo
        base_config["shielded_ratio"] = perc
        filenames.append(f"{algo}_{str(perc).replace('.','')}.json")
        with open(filenames[-1], "w") as f:
            s = json.dumps(base_config, indent=4)
            f.write(s)

runfile = f"""#!/bin/bash

#SBATCH --partition=genoa
#SBATCH --job-name=Debug_MultiAgent
#SBATCH --ntasks={len(filenames)*nseeds}
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --output=/home/schatterji1/Debug_MultiAgent/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate pls

cd $HOME/ShieldedMARL/pettingzoo

""" + \
" &\n".join([f"srun -n {nseeds} python main.py --config=configs/ma_msh/msh1/{filename}" for filename in filenames])

with open("run.job", "w") as f:
    f.write(runfile)
