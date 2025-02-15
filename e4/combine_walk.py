import os
import pathlib
import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

def output_gpx(points, output_filename):
    from xml.dom.minidom import getDOMImplementation
    xmlns = 'http://www.topografix.com/GPX/1/0'
    
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.10f' % (pt['latitude']))
        trkpt.setAttribute('lon', '%.10f' % (pt['longitude']))
        time = doc.createElement('time')
        time.appendChild(doc.createTextNode(pt['datetime'].strftime("%Y-%m-%dT%H:%M:%SZ")))
        trkpt.appendChild(time)
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    doc.documentElement.setAttribute('xmlns', xmlns)

    with open(output_filename, 'w') as fh:
        fh.write(doc.toprettyxml(indent='  '))

def get_data(input_gpx):
    gpx_file = input_gpx
    points = pd.DataFrame(columns=['latitude', 'longitude', 'datetime'])
    
    tree = ET.parse(gpx_file)
    root = tree.getroot()
    
    latitude = []
    longitude = []
    time_list = []
    
    for trkpt in root.iter('{http://www.topografix.com/GPX/1/0}trkpt'):
        lat = float(trkpt.get('lat'))
        latitude.append(lat)
        lon = float(trkpt.get('lon'))
        longitude.append(lon)
        time = trkpt.find('{http://www.topografix.com/GPX/1/0}time').text
        time_list.append(time)

    points['latitude'] = latitude
    points['longitude'] = longitude
    points['datetime'] = pd.to_datetime(time_list, utc=True)
    
    return points

def main():
    input_directory = pathlib.Path(sys.argv[1])
    output_directory = pathlib.Path(sys.argv[2])
    
    accl = pd.read_json(input_directory / 'accl.ndjson.gz', lines=True, convert_dates=['timestamp'])[['timestamp', 'x']]
    gps = get_data(input_directory / 'gopro.gpx')
    phone = pd.read_csv(input_directory / 'phone.csv.gz')[['time', 'gFx', 'Bx', 'By']]

    offset =0
    first_time = accl['timestamp'].min()
    phone['timestamp'] = first_time + pd.to_timedelta(phone['time']+offset, unit='sec')
    
    max_corr = -np.inf 
    best_offset = None 

    for offset in np.linspace(-5.0, 5.0, 101):
        phone['adjusted_timestamp'] = phone['timestamp'] + pd.to_timedelta(offset, unit='sec')
        phone['timestamp_rounded'] = phone['adjusted_timestamp'].dt.round('4S')
        phone_grouped = phone.groupby('timestamp_rounded')[['gFx']].mean()

        accl['timestamp_rounded'] = accl['timestamp'].dt.round('4S')
        accl_grouped = accl.groupby('timestamp_rounded')[['x']].mean()
        
        merged_data = pd.merge(accl_grouped, phone_grouped, left_index=True, right_index=True, how='inner')
        
        correlation = (merged_data['gFx'] * merged_data['x']).sum()
        
        if correlation > max_corr:
            max_corr = correlation
            best_offset = offset

    print(f"Best offset: {best_offset} and correlation is {max_corr}")
    
    # combined df
    phone['adjusted_timestamp'] = phone['timestamp'] + pd.to_timedelta(best_offset, unit='sec')
    phone['timestamp_rounded'] = phone['adjusted_timestamp'].dt.round('4S')
    
    phone_grouped = phone.groupby('timestamp_rounded')[['gFx', 'Bx', 'By']].mean()
    accl['timestamp_rounded'] = accl['timestamp'].dt.round('4S')
    accl_grouped = accl.groupby('timestamp_rounded')[['x']].mean()
    gps['timestamp_rounded'] = gps['datetime'].dt.round('4S')
    gps_grouped = gps.groupby('timestamp_rounded')[['latitude', 'longitude']].mean()
    
    combined = pd.merge(accl_grouped, phone_grouped, left_index=True, right_index=True, how='inner')
    combined = pd.merge(combined, gps_grouped, left_index=True, right_index=True, how='inner')
    
    combined.reset_index(inplace=True)
    combined.rename(columns={'timestamp_rounded': 'datetime'}, inplace=True)

    #outputed files
    output_gpx(combined[['datetime', 'latitude', 'longitude']], 'walk1/walk.gpx')
    combined[['datetime', 'Bx', 'By']].to_csv('walk1/walk.csv', index=False)

main()

