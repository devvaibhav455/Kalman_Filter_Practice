#! /usr/bin/env python3

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt


# Get the current script's file path
current_file_path = Path(__file__).resolve().parent
print(current_file_path)

filename = os.path.basename(__file__).split('.')[0] 

def main():
    
    # if len(sys.argv) < 2:
    #     print('Usage: python3 sensor_fusion.py 1 or 2')
    #     sys.exit(1)
    data = os.path.join(current_file_path, 'data', 'sample-laser-radar-measurement-data-1.txt' )
    # if sys.argv[1] == '1':
    #     data = os.path.join(current_file_path, '../data', 'sample-laser-radar-measurement-data-1.txt' )
    # elif sys.argv[1] == '2':
    #     data = os.path.join(current_file_path, '../data', 'sample-laser-radar-measurement-data-2.txt' )

    #Initialize the Kalman filter
    # State vector: [x, xdot, xddot, y, ydot, yddot]
    kf = KalmanFilter(dim_x=6, dim_z=4)
    dt = 1 #1 sec
    var = 0.1
    block_F = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
    kf.F[0:3, 0:3] = block_F
    kf.F[3:6, 3:6] = block_F
    # kf.H = #Converts the measurements to state space. Depending on sensor, change the H matrix
    block_H = np.array([[1, 0, 0], [0, 1, 0]])
    kf.H[0:2, 0:3] = block_H
    kf.H[2:4, 3:6] = block_H
    kf.P *= 10
    
    kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=var, block_size=2) #Dim is 3 because in state for one dimension, we have position, velocity, and acceleration and block_size is 2 because we have two of such dimensions i.e. x and y

    kf.x = np.array([[8], [0], [0], [0], [0], [0]]) #Initialize the sensor at (7,0) with 0 vel and acc
    x_vel, y_vel = 0, 0
    last_t = 1477010443299637 #0.1 seconds before the very first measurement
    
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [])
    

    try:
        with open(data, 'r') as file:
            for line in file:
                line = line.strip('\n')
                line = line.split('\t')
                
                #Need to update the R matrix depending on the sensor
                # R ρ φ ρ̇ timestamp x_gt y_gt vx_gt vy_gt
                # R 8.46642 0.0287602 -3.04035 1477010443399637 8.6 0.25 -3.00029 0
                if line[0] == 'R':
                    #Since this script using linear Kalman filter, we will transform the measurements in such a way that can be incorporated linearly into the H matrix
                    range, angle, angle_rate, time, x_gt, y_gt, vx_gt, vy_gt = float(line[1]), float(line[2]), float(line[3]), int(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8]) 
                    x_from_z = range*np.cos(angle)
                    y_from_z = range*np.sin(angle)
                    vx_from_z = angle_rate*np.cos(angle)
                    vy_from_z = angle_rate*np.sin(angle)
                    dt = (time - last_t)/1000000
                    last_t = time
                    var = 1
                    kf.R *= var 
                    z = np.array([[x_from_z, vx_from_z, y_from_z, vy_from_z]]).T

                # L x y timestamp x_gt y_gt vx_gt vy_gt
                elif line[0] == 'L':
                    x, y, time, x_gt, y_gt, vx_gt, vy_gt = float(line[1]), float(line[2]), int(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7])
                    #To fill the remaining readings in z for velocity, we take kf estimates from state and take difference with current readings
                    x_vel = x - kf.x[0][0]
                    y_vel = y - kf.x[3][0]
                    dt = (time - last_t)/1000000
                    last_t = time
                    var = 0.5
                    kf.R *= var
                    z = np.array([[x, x_vel, y, y_vel]]).T

                #Update the Q matrix since dt varies with the reading rate
                kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=var, block_size=2)
                print(line)
                print(kf.x[0][0], kf.x[3][0])
                kf.predict()
                kf.update(z)

                scatter.set_offsets([[kf.x[0][0], kf.x[3][0]]])
                ax.autoscale(True)
                ax.relim()
                # plt.pause(0.1)
    except Exception as e:
        print(f'Error occured: {e}')

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()