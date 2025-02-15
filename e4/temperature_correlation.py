import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

stations_filename = sys.argv[1]
city_data_filename = sys.argv[2]
output_name = sys.argv[3]

stations = pd.read_json(stations_filename, lines=True)
city=pd.read_csv(city_data_filename)

stations['avg_tmax']=stations['avg_tmax']/10

city=city.dropna(subset=['area', 'population'])
city['area']=city['area']/1000000
city = city[city['area'] <10000]
city['density'] = city['population'] / city['area']
#city

import numpy as np
import pandas as pd

def distance(city, stations):
    city_lat = np.radians(city['latitude'])
    city_lon = np.radians(city['longitude'])

    station_lat = np.radians(stations['latitude'].values)
    station_lon = np.radians(stations['longitude'].values)

    dlat = station_lat - city_lat
    dlon = station_lon - city_lon
    a = (np.sin(dlat / 2) ** 2 + 
         np.cos(city_lat) * np.cos(station_lat) * (np.sin(dlon / 2) ** 2))

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distances = 6371000 * c 
    
    return distances

def best_tmax(city, stations):
    distances = distance(city, stations)
    return stations.loc[np.argmin(distances), 'avg_tmax']

city['temperature'] = city.apply(best_tmax, axis=1, stations=stations)

plt.scatter(city['temperature'], city['density'])
plt.title('Temperature vs Population Density')
plt.xlabel('Avg Max Temperature (\u00b0C)')
plt.ylabel('Population Density (people/km\u00b2)')
plt.savefig(output_name)