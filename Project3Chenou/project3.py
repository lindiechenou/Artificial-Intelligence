# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 09:46:47 2020

@author: lindiechenou
"""

import pandas as pd
import numpy as np
import networkx as nx
#from timeit import default-timer
from timeit import default_timer


from math import  sqrt
# calculate the distance between two points
# approximate radius of earth in km
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
#calculate the distance between a line and a point
def dist_point_sect(x,y,z):
    
    x_lat = float(x[0])
    x_lon = float(x[1])
    y_lat = float(y[0])
    y_lon = float(y[1])
    z_lat = float(z[0])
    z_lon = float(z[1])
    if sqrt((z_lat - y_lat)**2 + (z_lon - y_lon)**2) != 0:
        distance_p = abs((z_lat - y_lat)*(y_lon - x_lon) - (y_lat - x_lat)*(z_lon - y_lon))/sqrt((z_lat - y_lat)**2 + (z_lon - y_lon)**2)
    else:
        distance_p = 0
    
    return distance_p
#Greedy algorith computationi
#this will compute the first node, then find the shortest path to the next node.
#with the line created the nearest node to the line will be calculated to the using the dist_point_sect
#then it will be visited
#then the remaining unvisited node will be visited,   
def selection_insertion(node,R_D, R_D_P):
    
    visited = [node]
    not_visited = set(range(R_D.shape[0])).difference(set(visited))
    weighted_not_visited = [(k,R_D[node][k]) for k in not_visited]
    next_node = min(weighted_not_visited, key=lambda yc: yc[1])
    visited.append(next_node[0])
    not_visited = set(range(R_D.shape[0])).difference(set(visited))
    #to add unvisited node to be visited
    while not_visited != set():
        weighted_not_visited = [(k,R_D[j][k]) for k in not_visited for j in visited]        
        next_node = min(weighted_not_visited, key=lambda yc: yc[1])
        insertion = [(k,k+1,R_D_P[next_node[0]][visited[k]][visited[k+1]]) for k in range(len(visited)-1)]
        inserted = min(insertion, key=lambda yc: yc[2])
        visited.insert(inserted[1],next_node[0])
        
        not_visited = set(range(R_D.shape[0])).difference(set(visited))
        #import pdb; pdb.set_trace()
    return visited
    #print(default_timer()-start)
    
        
    
def main():
    start = default_timer()
        #reading input from file
    cities = pd.read_table("Random30.tsp",delimiter = ' ',names=['rows_id','lat','lon'])[7:]
    df_cities = cities[['lat','lon']].reset_index().drop(['index'],axis=1)
    total_rows = cities.count().lat
     #defining the permition of the amount of input received
    #perm = permutations(range(0,total_rows))
    R_Dist = [[dist(df_cities.iloc[i],df_cities.iloc[j])  for i in range(total_rows)] for j in range(total_rows)]
    R_D = np.asarray(R_Dist)
    R_Dist_Point = {i:[[dist_point_sect(df_cities.iloc[i],df_cities.iloc[j],df_cities.iloc[k])  for j in range(total_rows)] for k in range(total_rows)] for i in range(total_rows)}
    
    #start = default_timer()
    Select = [(k,selection_insertion(k,R_D, R_Dist_Point),cost_path(selection_insertion(k,R_D, R_Dist_Point),R_D)) for k in range(R_D.shape[0])]
    G = min(Select,key=lambda yc: yc[2])[1]
    #print(default_timer()-start)
    #stop = timeit.default_timer()
    print(default_timer()-start)
    #creating the GUI depending on the amount of node given with the first node being display as green and the rest of the node being display as blue.
    edgelist = [(G[k],G[k+1]) for k in range(len(G)-1)]
    edgelist.append((G[len(G)-1],G[0]))
    graph = nx.OrderedDiGraph()   
    graph.add_edges_from(edgelist)
    color_map = []
    for node in graph.nodes:
        if node == G[0]:
            color_map.append('green')
        else: 
            color_map.append('blue')
    print(min(Select,key=lambda yc: yc[2]))
   
    

   # nx.draw_networkx(graph, node_color=color_map,with_labels=True)
    nx.draw_circular(graph, node_color=color_map,with_labels=True)
    #print(default_timer()-start)

    
    
    
if __name__ == "__main__":
    main()
        
