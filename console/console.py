class Console:
    @staticmethod
    def print_most_fitted(index, dna, value):
        print('Geração: {}\n --- Indivíduo mais apto: {}\n --- x = {:.4f}'.format(index, dna, value))

    @staticmethod
    def print_best_solution(x, fx):
        print('Melhor resultado:\n --- x = {}\n --- f(x) = {}'.format(x, fx))