from ga import GeneticAlgorithm, Population
from plot import PlotFrame
from console import Console
import numpy


def run(ga: GeneticAlgorithm) -> None:
    """
    Executa o algoritmo.
    :param ga: algoritmo genético a ser executado
    """

    # cria a população:
    population = Population(ga.get_population_size(), ga.get_bit_len())
    # cria a tela de plotagem do gráfico:
    plot = PlotFrame(ga.get_x_bounds(), ga.f)

    pause = input('PRESS ENTER TO START...')
    x_value = None

    for i in range(ga.get_num_generations()):
        # faz a plotagem dos dados:
        pop_float_values = ga.binary_to_float(population.get())
        plot.update(pop_float_values)
        plot.pause(0.1)
        if i < ga.get_num_generations() - 1:
            plot.clear()

        # evelução do algoritmo genético:
        fitness = ga.calculate_fitness(ga.f(pop_float_values))

        dna_fit = population.get_fitness(fitness)
        x_value = ga.binary_to_float(dna_fit)
        Console.print_most_fitted(i+1, dna_fit, x_value)

        population.set(ga.select(population.get(), fitness))
        pop_copy = population.copy()
        for parent in population.get():
            child = ga.crossover(parent, pop_copy)
            child = ga.mutate(child)
            parent[:] = child

    Console.print_best_solution(x_value, ga.f(x_value))

    plot.show()


if __name__ == '__main__':

    ga = GeneticAlgorithm(
        dna_len=8,
        pop_size=30,
        cross_rate=0.7,
        mutation_rate=0.01,
        n_generations=50,
        x_bound=(-10, 10),
        foo=lambda x: x**2 + 3*x - 14 #numpy.sin(10*x)*x + numpy.cos(2*x)*x
    )

    run(ga)
