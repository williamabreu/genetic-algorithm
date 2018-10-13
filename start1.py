# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:55:18 2017
@author: paulo
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

CROSSOVER_RATE = 0.9 #RATE OF CROSSOVER
MUTATION_RATE = 0.001 #RATE OF MUTATION
POP_SIZE = 100 #POPULATION SIZE
N_GENERATIONS = 1000 #MAXIMUM NUMBER OF GENERATIONS
CHRM_SIZE = 64 #CHROMOSSOME SIZE IN BITS
SCHRM_SIZE = CHRM_SIZE/2 #SIZE OF THE PART OF THE CHROMOSSOME THAT REPRSENTS X1
MIN_INTERVAL = -32.768
MAX_INTERVAL = 32.768

#creates a new population
def NewPop():
    return np.random.randint(0, 2, size=(POP_SIZE, CHRM_SIZE))

#converts a binary number represented in a array into an integer number
def BinToInt(nBin):
    return int(''.join(str(x) for x in nBin), 2)
#function f(x)
def f(x):
    return -20*np.exp(-0.2*np.sqrt((1.0/x.__len__())*sum([xi**2 for xi in x]))) - np.exp((1.0/x.__len__())*sum([np.cos(2*np.pi*xi) for xi in x])) + 20 + np.exp(1)
#function that receives a single chromossome an return its fitness
def Fitness(chrm):
    x1, x2 = ExtractX1X2(chrm)
    x = [x1, x2]
    return 1.0/(1+f(x))
def ExtractX1X2(chrm):
    ch1 = chrm[:(SCHRM_SIZE)]
    ch2 = chrm[(SCHRM_SIZE):]    
    x1 = MIN_INTERVAL+(MAX_INTERVAL-MIN_INTERVAL)*float(BinToInt(ch1))/((2**SCHRM_SIZE) - 1)
    x2 = MIN_INTERVAL+(MAX_INTERVAL-MIN_INTERVAL)*float(BinToInt(ch2))/((2**SCHRM_SIZE) - 1)
    return x1, x2
#function that returns the finesses of an entire population in the form of an array
def PopFitness(pop):
    return [Fitness(chrm) for chrm in pop]
#roullete
def RoulleteSelection(fitness):
    perc = np.array(fitness)/sum(fitness)
    total = 0.0
    for i in range(len(perc)):
        perc[i] += total
        total = perc[i]
    s = np.random.random()
    for i in range(len(perc)):
        if s <= perc[i]:
            return i
#crossing two chromossomes        
def Crossover(father, mother):
    limit = np.random.randint(CHRM_SIZE)
    return (list(father[:limit])+list(mother[limit:]),list(mother[:limit])+list(father[limit:]))
#changing a bit in a random position
def Mutation(chrm):
    pos = np.random.randint(CHRM_SIZE)
    chrm[pos] = 1 - chrm[pos]
    return chrm

solutions = []
for exe in range(0,30):
    
    pop = NewPop()
    best = []
    mean = []
    for gen in range(N_GENERATIONS):   
        fitness = PopFitness(pop)
        best.append(np.max(fitness))
        mean.append(np.mean(fitness))
        #<elitism>
        bestChrm = pop[fitness.index(best[-1])]
        #</elitism>
        nextPop = []    
        for i in range(POP_SIZE/2):
            father = pop[RoulleteSelection(fitness)]
            mother = pop[RoulleteSelection(fitness)]
            
            tx = np.random.random()
            if tx <= CROSSOVER_RATE:           
                son, daughter = Crossover(father, mother)
            else:
                son, daughter = father, mother
            tx = np.random.random()
            if tx <= MUTATION_RATE:
                son = Mutation(son)
            tx = np.random.random()
            if tx <= MUTATION_RATE:
                daughter = Mutation(daughter)
            nextPop.append(son)
            nextPop.append(daughter)
        pop = nextPop
        #elitism
        auxF = PopFitness(pop)
        pop[auxF.index(np.min(auxF))] = bestChrm
        #</elitism>
    fitness = PopFitness(pop)
    best.append(np.max(fitness))
    mean.append(np.mean(fitness))   
    
    solutions.append(best[-1])
    indexSol = fitness.index(solutions[-1])
    x1, x2 = ExtractX1X2(pop[indexSol])
    print "Solution: ("+str(x1)+", "+str(x2)+") with fitness: "+str(solutions[-1])
    
    x = np.arange(0.0, N_GENERATIONS+1, 1.0)
    plt.plot(x, mean)
    plt.plot(x, best)
    plt.show()

print "Mean of Solutions: "+str(np.mean(solutions))
print "Standard Deviation: "+str(np.std(solutions))

'''    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(MIN_INTERVAL/10, MAX_INTERVAL/10, 0.01)
y = x[:]
x, y = np.meshgrid(x, y)
z = x[:]+5
for i in range(0,x.__len__()):
    for j in range(0,x.__len__()):
        z[i][j] = 1.0/(1+f([x[i][j],y[i][j]]))
        #z[i][j] = f([x[i][j],y[i][j]])
ax.plot_surface(x, y, z, cmap=cm.coolwarm)
#ax.plot_wireframe(x, y, z, rstride=10, cstride=10)
#ax.scatter(x1, x2, bestFitness, c='g', marker='^')
#ax.text(x1, x2, bestFitness, "Best Point Found", color='green')
plt.show()
'''