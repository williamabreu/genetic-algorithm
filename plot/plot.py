import matplotlib.pyplot as matpltot
import numpy


# definição do tipo ponteiro de função
function = type(lambda: None)


class PlotFrame:
    """
    Tela com o gráfico da função.
    """

    def __init__(self, x_bounds: tuple, foo: function):
        """
        Cria a tela do gráfico.

        :param x_bounds: limites inferior e superior do eixo x
        :param foo: ponteiro da função para plotagem
        """
        matpltot.ion()
        x = numpy.linspace(*x_bounds, 200)
        matpltot.plot(x, foo(x))
        self.__scatter = None
        self.__x_bounds = x_bounds
        self.__foo = foo

    def update(self, pop_float_values: numpy.ndarray) -> None:
        """
        Atualiza os pontos no gráfico.
        :param pop_float_values: novos pontos para dispor
        """
        self.__scatter = matpltot.scatter(pop_float_values, self.__foo(pop_float_values), s=200, lw=0, c='red', alpha=0.5)

    def pause(self, seconds: float) -> None:
        """
        Pausa a execução.
        :param seconds: tempo
        """
        matpltot.pause(seconds)

    def clear(self) -> None:
        """
        Limpa os pontos no gráfico.
        """
        self.__scatter.remove()

    def show(self) -> None:
        """
        Chamado no final das mudanças na tela.
        """
        matpltot.ioff()
        matpltot.show()