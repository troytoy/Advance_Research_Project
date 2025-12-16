import random
import numpy as np
from deap import base, creator, tools, algorithms
from data_loader import load_data
from strategy import MACDStrategy

# 1. Setup Data
data = load_data()
strategy = MACDStrategy(data)

# 2. Setup GA
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attributes generator for MACD
# Fast: 5 to 50
toolbox.register("attr_fast", random.randint, 5, 50)
# Slow: 20 to 100
toolbox.register("attr_slow", random.randint, 20, 100)
# Signal: 5 to 50
toolbox.register("attr_signal", random.randint, 5, 50)

# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_fast, toolbox.attr_slow, toolbox.attr_signal), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3. Fitness Function
def evalMACD(individual):
    fast, slow, signal = individual
    # Constraint: Fast < Slow
    if fast >= slow:
        return -9999,
    
    fitness = strategy.evaluate(fast, slow, signal)
    return fitness,

toolbox.register("evaluate", evalMACD)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[5, 20, 5], up=[50, 100, 50], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga():
    random.seed(42)
    
    pop = toolbox.population(n=50)
    ngen = 10
    cxpb, mutpb = 0.5, 0.2
    
    print("Starting GA Optimization for MACD...")
    
    final_pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, 
                                           ngen=ngen, verbose=True)
    
    best_ind = tools.selBest(final_pop, 1)[0]
    print(f"Best Individual: Fast={best_ind[0]}, Slow={best_ind[1]}, Signal={best_ind[2]}")
    print(f"Best Fitness (Return): {best_ind.fitness.values[0]:.4f}")
    
    return best_ind[0], best_ind[1], best_ind[2]

if __name__ == "__main__":
    best_fast, best_slow, best_signal = run_ga()
