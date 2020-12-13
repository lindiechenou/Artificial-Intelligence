#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 19:15:42 2020

@author: lindiechenou
"""
from math import  sqrt
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
#formula to calculate the distance between nodes.
def dist_xy(x,y):
    x_lat = float(x[0])
    x_lon = float(x[1])
    y_lat = float(y[0])
    y_lon = float(y[1]) 
    
    
    distance = sqrt((y_lat-x_lat)**2 + (y_lon - x_lon)**2)
    return distance

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        
        distance = np.sqrt(((self.x-city.x)**2)+(self.y-city.y)**2)
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
#will be the inverse of the route distance. 
#since we want to minimize the distance we will a larger score to start with.    
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness    
#now to create an initial population we use the random function in the createRoute funtion
#then use the initial population loop around till we get the rest of the routes that can be populate.

def initialPopulation(popSize, cityList):
    def createRoute(cityList):
        route = random.sample(cityList, len(cityList))
        return route
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population
#now the routes will be assiciated with their distance scores
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
#now the creating of the parents.
#using the roulette wheel to selet which route ID to perform
#Parents are selected according to their fitness
#the better chromosomes they have the better their chances of being selected
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults
#with the selection of the routeID, now the mating pool can be created.
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool
#creation of the offspring population

def crossoverPopulation(matingpool, eliteSize):
    def crossover(parent1, parent2):
        child = []
        childP1 = []
        childP2 = []
        
        geneA = int(random.random() * len(parent1))
        geneB = int(random.random() * len(parent1))
        
        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)
    
        for i in range(startGene, endGene):
            childP1.append(parent1[i])
            
        childP2 = [item for item in parent2 if item not in childP1]
    
        child = childP1 + childP2
        return child
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = crossover(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children
#using the Swap Mutation on the Mutation Operators
#we swap two cities in our route with low probabilities
#this will run through the population

def mutatePopulation(population, mutationRate):
    def mutation(individual, mutationRate):
        for swapped in range(len(individual)):
            if(random.random() < mutationRate):
                swapWith = int(random.random() * len(individual))
                
                city1 = individual[swapped]
                city2 = individual[swapWith]
                
                individual[swapped] = city2
                individual[swapWith] = city1
        return individual
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutation(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop
#the NextGeneration funtion will just run everything that was created back againg.
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = crossoverPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

#we create the initial population, where we will loop through many generations.
#having the initial distance, after it's done looping through the final distance will displayed
def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Cost')
    plt.xlabel('Generation')
    plt.show()
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute
    
cities = pd.read_table("Random100.tsp",delimiter = ' ',names=['rows_id','lat','lon'])[7:]
cityList_np = cities.to_numpy().astype(float)[:,1:3]
cityList = []
for i in range(0,cityList_np.shape[0]):
    cityList.append(City(x=cityList_np[i][0], y=cityList_np[i][1]))
    
def plotPath(X, Y):
    plt.plot(X, Y, 'bo-')
    plt.plot(X[0], Y[0], 'rs-')
    plt.axis('scaled'); plt.axis('off')
    plt.show()

def main():
    #cities = pd.read_table("Random100.tsp",delimiter = ' ',names=['rows_id','lat','lon'])[7:]
    #cityList = cities.to_numpy().astype(float)[:,1:3]
    
    #geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
    #geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.001, generations=500)
    #geneticAlgorithmPlot(population=cityList, popSize=200, eliteSize=20, mutationRate=0.01, generations=500)
    geneticAlgorithmPlot(population=cityList, popSize=200, eliteSize=20, mutationRate=0.0001, generations=500)
    
    '''
    #to plot all of the result cities and their coordinates in a x and y table
    
    OptRoute = geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
    sX = []
    sY = []
    for i in range(len(OptRoute)):
        sX.append(OptRoute[i].x)
        sY.append(OptRoute[i].y)
    
    sX.append(OptRoute[0].x)
    sY.append(OptRoute[0].y)
    #plotPath(sX, sY)'''

if __name__ == "__main__":
    main()