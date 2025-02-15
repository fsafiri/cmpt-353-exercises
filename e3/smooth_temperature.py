import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter
import sys


input_file = sys.argv[1]
cpu_data = pd.read_csv(input_file)


cpu_data['timestamp'] = pd.to_datetime(cpu_data['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')


loess_smoothed =lowess(cpu_data['temperature'], cpu_data['timestamp'], frac=0.003)


kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]

initial_state = kalman_data.iloc[0]
observation_covariance = np.diag([3, 0.22, 0.98, 0.02])**2 
transition_covariance = np.diag([5, 0.25, 0.98, 0.04])**2
transition = [[0.97,0.5,0.2,0.001],[0.1,0.4,2.1,0],[0,0,0.94,0],[0,0,0,1]] 


kf = KalmanFilter(
    transition_matrices=transition,
    transition_covariance=transition_covariance,
    observation_covariance=observation_covariance,
    initial_state_mean=initial_state,
    initial_state_covariance=observation_covariance)

kalman_smoothed, _ = kf.smooth(kalman_data)

plt.figure(figsize=(12, 6))  
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5, label='Raw Temperature')


plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-', label='loess Smoothed')
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-', label='Kalman Smoothed')

plt.xticks(rotation=25)
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.title('CPU Temperature, LOESS Smoothed, and Kalman Smoothed')
plt.legend()
plt.savefig('cpu.svg')

print("done, figuresaved ")




