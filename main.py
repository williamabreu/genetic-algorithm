import matplotlib.pyplot as plt
import numpy as np
from ga import GeneticAlgorithm

def main(ga):
    population = np.random.randint(2, size=(ga.get_population_size(), ga.get_bit_len()))  # initialize the pop DNA

    plt.ion()  # something about plotting
    x = np.linspace(*ga.get_x_bounds(), 200)
    plt.plot(x, ga.f(x))

    for _ in range(ga.get_num_generations()-1):
        F_values = ga.f(ga.binary_to_float(population))  # compute function value by extracting DNA

        # something about plotting
        sca = plt.scatter(ga.binary_to_float(population), F_values, s=200, lw=0, c='red', alpha=0.5)
        plt.pause(0.1)
        sca.remove()

        # GA part (evolution)
        fitness = ga.calculate_fitness(F_values)
        dna_fit = population[np.argmax(fitness)]
        print("Most fitted DNA: ", dna_fit, ga.binary_to_float(dna_fit))
        population = ga.select(population, fitness)
        pop_copy = population.copy()
        for parent in population:
            child = ga.crossover(parent, pop_copy)
            child = ga.mutate(child)
            parent[:] = child  # parent is replaced by its child
    
    F_values = ga.f(ga.binary_to_float(population))  # compute function value by extracting DNA

    # something about plotting
    sca = plt.scatter(ga.binary_to_float(population), F_values, s=200, lw=0, c='red', alpha=0.5)

    # GA part (evolution)
    fitness = ga.calculate_fitness(F_values)
    dna_fit = population[np.argmax(fitness)]
    print("Most fitted DNA: ", dna_fit, ga.binary_to_float(dna_fit))
    population = ga.select(population, fitness)
    pop_copy = population.copy()
    for parent in population:
        child = ga.crossover(parent, pop_copy)
        child = ga.mutate(child)
        parent[:] = child  # parent is replaced by its child

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    ga = GeneticAlgorithm(
        DNA_LEN=8,
        POP_SIZE=30,
        CROSS_RATE=0.7,
        MUTATION_RATE=0.01,
        N_GENERATIONS=20,
        X_BOUND=[-10, 10],
        FUNCTION=lambda x: np.sin(10*x)*x + np.cos(2*x)*x
    )

    main(ga)
