class Console:
    @staticmethod
    def print_most_fitted(index, dna, value):
        print('Geração: {}\n --- Indivíduo mais apto: {}\n --- f(x) = {:.4f}'.format(index, dna, value))