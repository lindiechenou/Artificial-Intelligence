#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 18:41:04 2020

@author: lindiechenou
"""
#i code python project in anaconda python
#then open up spider where i can program and see the result of the program

#import heapq
from heapq import *
import pandas as pd
import numpy as np
from math import  sqrt
#method to calculate distance of this node to the other node
#using the distance formula
def dist(x,y):
    
    x_lat = float(x[0])
    x_lon = float(x[1])
    y_lat = float(y[0])
    y_lon = float(y[1]) 
    
    
    distance = sqrt((y_lat-x_lat)**2 + (y_lon - x_lon)**2)
    return distance
#calculate the cost between each city; for example 1-2,2-3,3-4
def cost_path(u:list, R_D):
    cost = 0
    if len(u) == 0: return None
    elif len(u) == 1: return 0
    else:
        cost += sum([R_D[u[k]][u[k+1]] for k in range(len(u)-1)])
        return cost
    
#function to drtaw the connected grapth between each node.
#to find all the possible path from node 0 to node 10.
#the connection between node to another.
def find_all_paths(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        #if not graph.has_key(start):
        #    return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths
#to calculate the shortes cost and path using DFS
def dfs(graph, source, destination, R_D):     
    frontier = []
    empty_list = []
    short_path = {}
    short_path[0] = 0
    heappush(frontier, 0)
    explored = []
    #neighborhood is representing children for each node.
    while frontier !=empty_list:
        current_node = heappop(frontier)
        if current_node in explored: continue
        if current_node == destination: break
        neighborhood = list(graph.get(current_node))
        cost_v = []
        #jit's going to explore all children and their children until it come to a stop.
        #after that goes to the parent node other children and see if they have been visited or not on that path.
        for v in neighborhood:            
            for u in find_all_paths(graph, source, v, path=[]):                
                cost_v.append((v, u,cost_path(u, R_D)))
            min_cost = min(cost_v, key=lambda yc: yc[2])
        short_path = (min_cost[1], min_cost[2]) 
        explored.append(current_node)
        heappush(frontier,min_cost[0])
    return  short_path
#to calculate the shortes cost and path using BFS
def bfs(graph, source, destination, R_D):    
    frontier = []
    empty_list = []
    short_path = {}
    short_path[0] = 0
    heappush(frontier, 0)
    explored = set([source])
    while frontier !=empty_list:
        current_node = heappop(frontier)
        
        if current_node == destination: break
    #neighborhood is representing children for each node.
        neighborhood = list(graph.get(current_node))
        cost_v = []
        #visit all children and put them in a queue 
        #in which after all parent children node have been visited
        #then make the next children node the new parent node.
        for v in neighborhood:
            if v not in explored:
                explored.add(v)
                for u in find_all_paths(graph, source, v, path=[]):
                    cost_v.append((v, u,cost_path(u, R_D)))
                min_cost = min(cost_v, key=lambda yc: yc[2])
        heappush(frontier,min_cost[0])
        short_path = (min_cost[1], min_cost[2])    
    return  short_path
def main():
    #reading input from file
    cities = pd.read_table("11PointDFSBFS.tsp",delimiter = ' ',names=['rows_id','lat','lon'])[7:]
    df_cities = cities[['lat','lon']].reset_index().drop(['index'],axis=1)
    total_rows = cities.count().lat
    #defining the permition of the amount of input received
    #perm = permutations(range(0,total_rows))
    R_Dist = [[dist(df_cities.iloc[i],df_cities.iloc[j])  for i in range(total_rows)] for j in range(total_rows)]
    RD = np.asarray(R_Dist)
    graph = {0:[1,2,3], 1:[2], 2:[3,4], 3:[4,5,6], 4:[6,7], 5:[7], 6:[8,9], 7:[8,9,10], 8:[10], 9:[10], 10:[]}
    #print("BFS")
    #print(bfs(graph, 0, 10, RD))
    print("DFS")
    print(dfs(graph, 0, 10, RD))
                      

if __name__ == "__main__":
    main()                                 
