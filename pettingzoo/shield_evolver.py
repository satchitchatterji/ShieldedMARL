import sys
sys.path.append("..")
import os

import random
from deap import base, creator, tools
import grape.grape as grape  # GRAPE module (ensure grape.py is available)
import copy

# Import ProbLog libraries
from problog.program import PrologString
from problog import get_evaluatable

class ShieldEvolver:

    def __init__(self, 
                 env_name,
                 fitness_fn, 
                 grammar_file, 
                 sensor_names,
                 action_names,
                 pop_size=100, 
                 n_gen=10, 
                 cx_pb=0.7, 
                 mut_pb=0.3, 
                 random_seed=42, 
                 verbose=False,
                 ):
        self.POP_SIZE = pop_size              # population size
        self.N_GEN = n_gen                 # number of generations to evolve
        self.CX_PB = cx_pb                # crossover probability
        self.MUT_PB = mut_pb               # mutation probability
        self.RANDOM_SEED = random_seed           # seed for reproducibility (optional)
        random.seed(self.RANDOM_SEED)
        self.env_name = env_name
        self.sensor_names = sensor_names
        self.action_names = action_names
        self.shield_skeleton = self.generate_shield_skeleton()
        self.verbose = verbose

        self.GRAMMAR_FILE = grammar_file
        self.BNF_GRAMMAR = grape.Grammar(self.GRAMMAR_FILE)  # load the ProbLog grammar
        self.toolbox = base.Toolbox()

        self.fitness_fn = fitness_fn
        self.setup_shield_evolution()
        self.shield_files = {}
        self.population = self.init_population()
        self.base_dir = f"evolve/shields/{self.env_name}"
        os.makedirs(self.base_dir, exist_ok=True)

        self.save_population(self.population)
        self.best_ind = None
        self.best_fitness = None

    def generate_shield_skeleton(self):
        sensor_str = ""
        for idx, sensor in enumerate(self.sensor_names):
            sensor_str += f"sensor({idx})::sensor({sensor}).\n"
        action_str = ""
        for idx, action in enumerate(self.action_names):
            action_str += f"action({idx})::action({action});\n"
        action_str = action_str[:-2] + ".\n"
        safe_str = "safe_next :- \+unsafe_next.\n"

        shield_skeleton = "\n".join([sensor_str, action_str, safe_str])+"\n"
        # print(shield_skeleton)
        # exit()
        return shield_skeleton

    def save_population(self, population):
        for i, ind in enumerate(population):
            with open(f"{self.base_dir}/{i}.pl", "w") as f:
                phenotype = str(ind.phenotype).replace(". ", ". \n")
                f.write(self.shield_skeleton + phenotype)
            self.shield_files[ind] = f"{self.base_dir}/{i}.pl"

    def save_single_ind(self, ind, name):
        with open(f"{self.base_dir}/{name}.pl", "w") as f:
            phenotype = str(ind.phenotype).replace(". ", ". \n")
            f.write(self.shield_skeleton + phenotype)
        self.shield_files[ind] = f"{self.base_dir}/{name}.pl"

    def setup_shield_evolution(self):
        # **2. Set up DEAP individuals and fitness using GRAPE's Individual**
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", grape.Individual, fitness=creator.FitnessMax)

        # Use GRAPE's initialisation method (sensible_initialisation = ramped half-and-half)
        self.toolbox.register("population_creator", grape.sensible_initialisation, creator.Individual)

        # **3. Register genetic operators**
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", grape.crossover_onepoint)
        self.toolbox.register("mutate", grape.mutation_int_flip_per_codon)
        self.toolbox.register("evaluate", self.fitness_fn)

    def init_population(self):
        # **5. Initialize population**
        population = self.toolbox.population_creator(
            pop_size=self.POP_SIZE,
            bnf_grammar=self.BNF_GRAMMAR,
            min_init_depth=10,      # minimum initial derivation-tree depth (ensures multiple clauses)
            max_init_depth=30,      # maximum initial derivation-tree depth
            codon_size=255,         # codon value range for genome (0-255)
            codon_consumption='lazy', 
            genome_representation='list'
        )

        return population

    def evaluate_population(self, population):
        # Evaluate initial population
        for ind in population:
            ind.fitness.values = self.toolbox.evaluate(ind)

        return population
    
    def evolve_single_gen(self):
        # Select parents for the next generation
        selected = self.toolbox.select(self.population, len(self.population))
        # Clone the selected individuals (to create offspring population)
        offspring = [copy.deepcopy(ind) for ind in selected]

        # Apply crossover on the offspring
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < self.CX_PB:
                offspring[i], offspring[i+1] = self.toolbox.mate(
                    offspring[i], offspring[i+1],
                    bnf_grammar=self.BNF_GRAMMAR,
                    max_depth=20,
                    codon_consumption='lazy'
                )
                # Invalidate fitness values of offspring after crossover
                del offspring[i].fitness.values, offspring[i+1].fitness.values

        # Apply mutation on the offspring
        for i in range(len(offspring)):
            if random.random() < self.MUT_PB:
                offspring[i], = self.toolbox.mutate(
                    offspring[i],
                    mut_probability=0.1,
                    bnf_grammar=self.BNF_GRAMMAR,
                    max_depth=20,
                    codon_consumption='lazy',
                    codon_size=255
                )
                del offspring[i].fitness.values

        # Evaluate new offspring (only those with invalid fitness)
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = self.toolbox.evaluate(ind)

        # Replace population with offspring (next generation)
        self.population = offspring
        self.save_population(self.population)

        # Find and print the best individual of this generation
        best_ind = max(self.population, key=lambda ind: ind.fitness.values[0])
        best_fitness = best_ind.fitness.values[0]

        self.best_ind = best_ind
        self.best_fitness = best_fitness

        # save best individual
        with open(f"{self.base_dir}/best_individual.pl", "w") as f:
            f.write(str(best_ind.phenotype).replace(". ", ". \n"))
        # save full shield
        self.save_single_ind(best_ind, f"best_individual_{best_fitness:.3f}")
    
    def evolve(self):
        for gen in range(1, self.N_GEN + 1):
            self.evolve_single_gen()
            if self.verbose:
                print(f"Generation {gen}: Best fitness = {self.best_fitness:.3f}")
                print("Best program:\n", str(self.best_ind.phenotype).replace(". ", ". \n"))
                print("-" * 40)

        # **7. (Optional) Final result**
        best_ind = max(self.population, key=lambda ind: ind.fitness.values[0])
        self.best_ind = best_ind
        self.best_fitness = best_ind.fitness.values[0]
        if self.verbose:
            print("==" * 20)
            print(f"Final best individual (fitness {best_ind.fitness.values[0]:.3f}):")
            print(str(best_ind.phenotype).replace(". ", ". \n"))
