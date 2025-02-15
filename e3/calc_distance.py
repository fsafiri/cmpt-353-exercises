import xml.etree.ElementTree as ET
import pandas as pd 
import numpy as np
from pykalman import KalmanFilter
import sys


gpx_file = sys.argv[1] 
csv_file = sys.argv[2]

points=pd.DataFrame(columns=['latitude','longitude','time'])

tree = ET.parse(gpx_file)
root = tree.getroot()
latitude=[]
longitude=[]
time_list=[]

for trkpt in root.iter('{http://www.topografix.com/GPX/1/0}trkpt'):
    
    lat = float(trkpt.get('lat'))
    latitude.append(lat)
    lon = float(trkpt.get('lon'))
    longitude.append(lon)
    time = trkpt.find('{http://www.topografix.com/GPX/1/0}time').text
    time_list.append(time)


points=pd.DataFrame()

points['latitude']=latitude
points['longitude']=longitude
points['datetime']=time_list


points['datetime'] = pd.to_datetime(points['datetime'], utc=True)


points = points.set_index('datetime')
sensor_data = pd.read_csv(csv_file, parse_dates=['datetime']).set_index('datetime')
points['Bx'] = sensor_data['Bx']
points['By'] = sensor_data['By']

points['lat_next'] = points['latitude'].shift(-1)
points['lon_next'] = points['longitude'].shift(-1)
points=points.dropna()

import math

def haversine_dist(row):

    lat1 = math.radians(row['latitude'])
    lat2 = math.radians(row['lat_next'])
    lon1 = math.radians(row['longitude'])
    lon2 = math.radians(row['lon_next'])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (math.sin((lat2 - lat1) / 2)) ** 2 + math.cos(lat1) * math.cos(lat2) * (math.sin((lon2 - lon1) / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    r = 6371000
    distance = r*c

    return distance


points['distance'] = points.apply(haversine_dist, axis=1)

column_sum = points['distance'].sum().round(2)

print(f"Unfiltered distance: {column_sum:.2f}")


import numpy as np
from pykalman import KalmanFilter


initial_state = points[['latitude', 'longitude', 'Bx', 'By']].iloc[0]

observation_covariance = np.diag([0.0001, 0.009, 5, 5]) ** 2  
transition_covariance = np.diag([0.0002, 0.01,5, 5]) ** 2 

transition = np.array([[1, 5 * 10 ** (-7), 34 * 10 ** (-7), 0],  
                        [0, 1, -49 * 10 ** (-7), 9 * 10 ** (-7)],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

kf = KalmanFilter(
    transition_matrices=transition,
    transition_covariance=transition_covariance,
    observation_covariance=observation_covariance,
    initial_state_mean=initial_state,
    initial_state_covariance=observation_covariance)

kalman_smoothed, _ = kf.smooth(points[['latitude', 'longitude', 'Bx', 'By']])

smoothed_df = pd.DataFrame(kalman_smoothed, columns=['latitude', 'longitude', 'Bx', 'By'])
smoothed_df['lat_next'] = smoothed_df['latitude'].shift(-1)
smoothed_df['lon_next'] = smoothed_df['longitude'].shift(-1)
smoothed_df['distance'] = smoothed_df.apply(haversine_dist,axis=1)

filtered_distance = smoothed_df['distance'].sum().round(2)

print(f"filtered distance: {filtered_distance:.2f} ")


"""
def output_gpx(points, output_filename):
   
    #Output a GPX file with latitude and longitude from the points DataFrame.
    
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.7f' % (pt['latitude']))
        trkpt.setAttribute('lon', '%.7f' % (pt['longitude']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


output_gpx(smoothed_df, 'kalman_smoothed_walk.gpx')
"""

