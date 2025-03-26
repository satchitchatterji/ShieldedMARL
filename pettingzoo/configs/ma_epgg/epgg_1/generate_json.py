import json

nseeds = 5
ncpus = 1
time = "01:00:00"
n_agents = 5
partition = "genoa"

with open('template.json') as f:
    base_config = json.load(f)

algos_big = ["SIPPO", "SMAPPO", "SACSPPO"]
percs = [round(n/n_agents,3) for n in range(n_agents+1)]
mus = [0.5, 1, 1.5, 2.5, 5]

for algo_big in algos_big:
    algos = [algo_big]
    filenames = []
    for algo in algos:
        for perc in percs:
            for mu in mus:
                base_config["algo"] = algo
                base_config["shielded_ratio"] = perc
                base_config["env_config"]["f_params"] = [mu,1]
                filenames.append(f"{algo}_{str(perc).replace('.','')}_{mu}.json")
                with open(filenames[-1], "w") as f:
                    s = json.dumps(base_config, indent=4)
                    f.write(s)

    runfile = f"""#!/bin/bash

#SBATCH --partition={partition}
#SBATCH --job-name=EPGG
#SBATCH --ntasks={len(filenames)*nseeds}
#SBATCH --cpus-per-task={ncpus}
#SBATCH --time={time}
#SBATCH --output=/home/schatterji1/EPGG/slurm_output_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate pls

cd $HOME/ShieldedMARL/pettingzoo

wandb offline

""" + \
" &\n".join([f"srun -n {nseeds} python main.py --config=configs/ma_epgg/epgg_1/{filename}" for filename in filenames])

    with open(f"../../../run_{algo_big}.job", "w") as f:
        f.write(runfile)

    print("Total run count:   ", len(filenames)*nseeds)
    print("Total cpu count:   ", len(filenames)*nseeds*ncpus)
    print("Total config count:", len(filenames))
    print("Node ref rome (128), genoa (192):", len(filenames)*nseeds*ncpus/128, len(filenames)*nseeds*ncpus/192)
    print("Node ref rome (16), genoa (16):", len(filenames)*nseeds*ncpus/16, len(filenames)*nseeds*ncpus/16)

print("\n")
print("n_nodes | rome | genoa")
print("   1    | 128  | 192")
print("   2    | 256  | 384")
print("   3    | 384  | 576")
print("   4    | 512  | 768")
print("   5    | 640  | 960")
print("   6    | 768  | 1152")
print("   7    | 896  | 1344")
print("   8    | 1024 | 1536")