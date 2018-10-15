import numpy


class Population:
    """
    Aglomerado de indivíduos do algoritmo.
    """

    def __init__(self, population_size: int, dna_len: int):
        """
        Cria a população aleatoriamente.

        :param population_size: quantidade de indivíduos
        :param dna_len: comprimento da cadeia do DNA
        """
        self.__individuals = numpy.random.randint(2, size=(population_size, dna_len))

    def get(self) -> numpy.ndarray:
        """
        Obtém o array de indivíduos.
        :return: todos os indivíduos da população
        """
        return self.__individuals

    def set(self, value: numpy.ndarray) -> None:
        """
        Altera os indivíduos da população.
        :param value: novo array de indivíduos
        """
        self.__individuals = value

    def copy(self) -> numpy.ndarray:
        """
        Copia o array de indivíduos.
        :return: cópia de todos os indivíduos da população
        """
        return self.__individuals.copy()