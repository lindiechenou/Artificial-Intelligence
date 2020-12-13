# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from itertools import permutations


#from math import sin, cos, sqrt, atan2, radians
#from math import radians

# approximate radius of earth in km
def dist(x,y):
    R = 1
    #R = 6373.0
    #define the x&y values for the nodes
    x_lat = float(x[0])
    x_lon = float(x[1])
    y_lat = float(y[0])
    y_lon = float(y[1])
  #  lat1 = radians(x_lat)
   # lon1 = radians(x_lon)
  #  lat2 = radians(y_lat)
   # lon2 = radians(y_lon)
    
   
    #method to calculate distance of this node to the other node
    #using the distance formula
    dlat = y_lat - x_lat
    dlon = y_lon - x_lon
    #dlon = lon2 - lon1
    #dlat = lat2 - lat1
    
    
    
    d = (dlat**2 + dlon**2)**0.5
    
   # a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    #c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * d
    return distance
def main():
   #reading input from file
    cities = pd.read_table("Random8.tsp",delimiter = ' ',names=['rows_id','lat','lon'])[7:]
    df_cities = cities[['lat','lon']].reset_index().drop(['index'],axis=1)
    total_rows = cities.count().lat
    #defining the permition of the amount of input received
    perm = permutations(range(0,total_rows))
    R_Dist = [[dist(df_cities.iloc[i],df_cities.iloc[j])  for i in range(total_rows)] for j in range(total_rows)]
    R_D = np.asarray(R_Dist)
    
    #import pdb; pdb.set_trace()
   #function to calculate the cost of each permitation
   #at the end of the calculation the least path will be given
    opt_path = tuple(range(total_rows))
    min_cost = float('inf')
    
    for per in perm:
        cost = 0
        cost += R_D[per[total_rows-1]][per[0]]
        for iter in range(total_rows-1):            
            cost += R_D[per[iter]][per[iter+1]]
        if cost < min_cost:
            opt_path = per
            min_cost = cost
    #import pdb; pdb.set_trace()
        
    print(min_cost)
    print(opt_path)
    
if __name__ == "__main__":
    main()
    