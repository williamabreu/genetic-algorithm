import numpy as np


class GeneticAlgorithm:
    def __init__(self, dna_len, pop_size, cross_rate, mutation_rate, n_generations, x_bound, function):
        self.__DNA_LEN = dna_len  # DNA length
        self.__POP_SIZE = pop_size  # population size
        self.__CROSS_RATE = cross_rate  # mating probability (DNA crossover)
        self.__MUTATION_RATE = mutation_rate  # mutation probability
        self.__N_GENERATIONS = n_generations  #
        self.__X_BOUND = x_bound  # x upper and lower bounds
        self.__FUNCTION = function  # to find the maximum of this function

    def get_bit_len(self):
        return self.__DNA_LEN

    def get_population_size(self):
        return self.__POP_SIZE

    def get_cross_rate(self):
        return self.__CROSS_RATE

    def get_mutation_rate(self):
        return self.__MUTATION_RATE

    def get_num_generations(self):
        return self.__N_GENERATIONS

    def get_x_bounds(self):
        return self.__X_BOUND

    def f(self, x):
        return self.__FUNCTION(x)

    # def set_bit_len(self, value):
    #     self.__DNA_LEN = value

    # def set_population_size(self, value):
    #     self.__POP_SIZE = value

    # def set_cross_rate(self, value):
    #     self.__CROSS_RATE = value

    # def set_mutation_rate(self, value):
    #     self.__MUTATION_RATE = value

    # def set_num_generations(self, value):
    #     self.__N_GENERATIONS = value

    # def set_x_bounds(self, value):
    #     self.__X_BOUND = value

    # def set_function(self, function):
    #     self.__FUNCTION = function

    # find non-zero fitness for selection
    def calculate_fitness(self, pred):
        return pred + 1e-3 - np.min(pred)

    # convert binary DNA to decimal and normalize it to a range(0, 5)
    def binary_to_float(self, pop):
        return (pop.dot(2 ** np.arange(self.__DNA_LEN)[::-1]) / float(2**self.__DNA_LEN-1) * (self.__X_BOUND[1] - self.__X_BOUND[0])) + self.__X_BOUND[0]

    def select(self, pop, fitness):  # nature selection wrt pop's fitness
        idx = np.random.choice(np.arange(self.__POP_SIZE), size=self.__POP_SIZE, replace=True,
                               p=fitness / fitness.sum())
        return pop[idx]

    def crossover(self, parent, pop):  # mating process (genes crossover)
        if np.random.rand() < self.__CROSS_RATE:
            i_ = np.random.randint(0, self.__POP_SIZE, size=1)  # select another individual from pop
            cross_points = np.random.randint(0, 2, size=self.__DNA_LEN).astype(np.bool)  # choose crossover points
            parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
        return parent

    def mutate(self, child):
        for point in range(self.__DNA_LEN):
            if np.random.rand() < self.__MUTATION_RATE:
                child[point] = 1 if child[point] == 0 else 0
        return child
