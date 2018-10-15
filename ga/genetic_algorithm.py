from builtins import enumerate

import numpy


# definição do tipo ponteiro de função
function = type(lambda: None)


class GeneticAlgorithm:
    """
    Maximiza uma função.
    """

    def __init__(self, dna_len: int, pop_size: int, cross_rate: float, mutation_rate: float, n_generations: int,
                 x_bound: tuple, foo: function):
        """
        Cria um novo algoritmo genético para encontrar o máximo da função.

        :param dna_len: quantidade de bits do DNA binário, quanto maior melhor a aproximação da função (resolução)
        :param pop_size: tamanho de indivíduos da população, quanto maior melhor será a diversidade
        :param cross_rate: probabilidade de crossover, entre 0 e 1
        :param mutation_rate: probabiliadade de mutação, entre 0 e 1
        :param n_generations: número de iterações para resolver o problema
        :param x_bound: limites inferior e superior do eixo x da função
        :param foo: ponteiro para a função objetivo a ser maximizada
        """
        if type(dna_len) != int or dna_len < 1:
            raise ValueError('dna_len is wrong')
        if type(pop_size) != int or pop_size < 1:
            raise ValueError('pop_size is wrong')
        if type(cross_rate) != float or not (0.0 <= cross_rate <= 1.0):
            raise ValueError('cross_rate is wrong')
        if type(mutation_rate) != float or not (0.0 <= mutation_rate <= 1.0):
            raise ValueError('mutation_rate is wrong')
        if type(n_generations) != int or n_generations < 1:
            raise ValueError('n_generations is wrong')
        if type(x_bound) != tuple or len(x_bound) != 2 or x_bound[0] >= x_bound[1]:
            raise ValueError('x_bound is wrong')
        if type(foo) != function:
            raise ValueError('foo is wrong')

        self.__DNA_LEN = dna_len
        self.__POP_SIZE = pop_size
        self.__CROSS_RATE = cross_rate
        self.__MUTATION_RATE = mutation_rate
        self.__N_GENERATIONS = n_generations
        self.__X_BOUND = x_bound
        self.__FUNCTION = foo

    def get_bit_len(self) -> int:
        """
        Obtém comprimento do DNA binário.
        :return: comprimento de bits do DNA binário
        """
        return self.__DNA_LEN

    def get_population_size(self) -> int:
        """
        Obtém o tamanho da população.
        :return: tamanho da população
        """
        return self.__POP_SIZE

    def get_cross_rate(self) -> float:
        """
        Obtém a taxa de crossover.
        :return: taxa de crossover
        """
        return self.__CROSS_RATE

    def get_mutation_rate(self)-> float:
        """
        Obtém a taxa de mutação.
        :return: taxa de mutação
        """
        return self.__MUTATION_RATE

    def get_num_generations(self) -> int:
        """
        Obtém número de gerações do algoritmo.
        :return: número de gerações
        """
        return self.__N_GENERATIONS

    def get_x_bounds(self) -> tuple:
        """
        Obtém os limites do eixo x.
        :return: tupla ordenada com os limites inferior e superior
        """
        return self.__X_BOUND

    def f(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Calcula o valor da função objetivo no ponto x (suporta numpy multiarray).

        :param x: índice do eixo x do ponto
        :return: valor da função em x
        """
        return self.__FUNCTION(x)

    def binary_to_float(self, pop_bin_values: numpy.ndarray) -> numpy.ndarray:
        """
        Converte para float o valor representado pelo DNA binário (suporta numpy multiarray).

        :param pop_bin_values: valores binários de DNA dos indivíduos
        :return: valores float de x dos indivíduos
        """
        population = pop_bin_values
        DNA_LEN = self.__DNA_LEN
        MAX_DNA_VALUE = 2 ** DNA_LEN - 1
        X_RANGE_LEN = self.__X_BOUND[1] - self.__X_BOUND[0]
        TRANSLATION = self.__X_BOUND[0]
        return population.dot(2 ** numpy.arange(DNA_LEN)[::-1]) / MAX_DNA_VALUE * X_RANGE_LEN + TRANSLATION

    def calculate_fitness(self, pop_float_values: numpy.ndarray) -> numpy.ndarray:
        """
        Calcula a aptidão dos indivíduos da população (suporta numpy multiarray).

        :param pop_float_values: array com valores calculados de f(x) para os indivíduos
        :return: array com os valores de aptidão dos indivíduos
        """
        return pop_float_values + 0.01 - numpy.min(pop_float_values)

    def select(self, pop_bin_values: numpy.ndarray, fitness: numpy.ndarray) -> numpy.ndarray:
        """
        Faz a seleção do indivíduo com maior aptidão da população (suporta numpy multiarray).

        :param pop_bin_values: valores binários de DNA dos indivíduos
        :param fitness: array com os valores de aptidão dos indivíduos
        :return: população selecionada pelo torneio
        """
        POP_SIZE = self.__POP_SIZE
        i = numpy.random.choice(numpy.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness/fitness.sum())
        return pop_bin_values[i]

    def crossover(self, parent: numpy.ndarray, pop_bin_values: numpy.ndarray) -> numpy.ndarray:
        """
        Aplica crossover aos genes dos indivíduos (suporta numpy multiarray).

        :param parent: valor binário de DNA do indivíduo pai
        :param pop_bin_values: valores binários de DNA dos indivíduos
        :return: filho gerado pelo crossover
        """
        CROSS_RATE = self.__CROSS_RATE
        POP_SIZE = self.__POP_SIZE
        DNA_LEN = self.__DNA_LEN
        if numpy.random.rand() < CROSS_RATE:
            # Seleciona aleatoriamente um indivíduo para o cruzamento:
            i = numpy.random.randint(0, POP_SIZE, size=1)
            # Marca aleatoriamente quais genes serão trocados entre os dois indivíudos:
            cross_genes = numpy.random.randint(0, 2, DNA_LEN).astype(numpy.bool)
            # Faz a troca do genes, cruzamento, entre os dois, gerando 1 filho:
            parent[cross_genes] = pop_bin_values[i, cross_genes]
        # retorna o filho:
        return parent

    def mutate(self, child: numpy.ndarray) -> numpy.ndarray:
        """
        Aplica mutação aos genes dos filhos dos indivíduos (suporta numpy multiarray).

        :param child: filho após ser gerado por crossover
        :return: filho após sofrer mutação
        """
        DNA_LEN = self.__DNA_LEN
        MUTATION_RATE = self.__MUTATION_RATE
        for gene in range(DNA_LEN):
            if numpy.random.rand() < MUTATION_RATE:
                # inverte o bit
                child[gene] = 1 if child[gene] == 0 else 0
        return child
