import tkinter

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# Implement the default Matplotlib key bindings.
from matplotlib.figure import Figure

import numpy as np


class Gui(tkinter.Tk):
    def __init__(self, ploting):
        super().__init__()
        self.wm_title("Embedding in Tk")
        self.geometry('800x600')
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.fig.add_subplot(111).plot(var, function)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        self.button = tkinter.Button(master=self, text="Quit", command=self.quit)
        self.button.pack(side=tkinter.BOTTOM)
    
if __name__ == '__main__':
    gui = Gui()
    gui.mainloop()