import sys
import os
import numpy as np
from pathlib import Path
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QCheckBox, QHBoxLayout
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

# Load data
current_file_path = Path(__file__).resolve().parent
data_path = os.path.join(current_file_path, 'data', 'sample-laser-radar-measurement-data-1_copy.txt')
# data_path = os.path.join(current_file_path, 'data', 'sample-laser-radar-measurement-data-2.txt')

def load_data(filepath):
    radar_data = []
    lidar_data = []
    with open(filepath, 'r') as file:
        last = None
        for line in file:
            line = line.strip().split('\t')
            # R ρ φ ρ̇ timestamp x_gt y_gt vx_gt vy_gt
            # R 8.46642 0.0287602 -3.04035 1477010443399637 8.6 0.25 -3.00029 0
            if line[0] == 'R':
                range_, angle, angle_rate, time, *_ = map(float, line[1:])
                x = range_ * np.cos(angle)
                y = range_ * np.sin(angle)
                vx = angle_rate * np.cos(angle)
                vy = angle_rate * np.sin(angle)
                radar_data.append((line[0], time, np.array([x, vx, y, vy]).reshape(4,1)))
            # L x y timestamp x_gt y_gt vx_gt vy_gt
            elif line[0] == 'L':
                x, y, time, *_ = map(float, line[1:])
                if last == None:
                    lidar_data.append((line[0], time, np.array([x, 0, y, 0]).reshape(4,1)))
                    last = x, y, time
                else:
                    vx = (x - last[0])/(time - last[2])*1e6
                    vy = (y - last[1])/(time - last[2])*1e6
                    lidar_data.append((line[0], time, np.array([x, vx, y, vy]).reshape(4,1)))

    return radar_data, lidar_data

radar_data, lidar_data = load_data(data_path)

class KalmanApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Real-Time Kalman Filter Tuning')
        self.resize(800, 600)

        layout = QVBoxLayout()

        # Plot widget
        self.plot = pg.PlotWidget()
        self.plot.setXRange(-10, 10)
        self.plot.setYRange(-10, 10)
        self.plot.addLegend()
        layout.addWidget(self.plot)

        # Timer for delayed updates
        self.update_timer = QTimer()
        self.update_timer.setInterval(100)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_kalman)
        self.var_scale = int(1e8)
        # Sliders and labels
        self.q_slider, self.q_label = self.create_slider('Process Noise (Q)', layout, 1, int(0.25*self.var_scale), default_value=int(0.000001*self.var_scale))
        self.radar_slider, self.radar_label = self.create_slider('Radar Noise (R)', layout, 1, 1*self.var_scale, default_value=int(0.003294*self.var_scale))
        self.lidar_slider, self.lidar_label = self.create_slider('Lidar Noise (R)', layout, 1, 1*self.var_scale, default_value=int(0.001891*self.var_scale))

        # Checkboxes
        self.radar_checkbox = QCheckBox('Use Radar')
        self.radar_checkbox.setChecked(True)
        self.radar_checkbox.stateChanged.connect(self.update_kalman)
        layout.addWidget(self.radar_checkbox)

        self.lidar_checkbox = QCheckBox('Use Lidar')
        self.lidar_checkbox.setChecked(True)
        self.lidar_checkbox.stateChanged.connect(self.update_kalman)
        layout.addWidget(self.lidar_checkbox)

        self.setLayout(layout)

        # Data curves
        self.radar_curve = self.plot.plot([], [], pen='r', name='Radar')
        self.lidar_curve = self.plot.plot([], [], pen='g', name='Lidar')
        self.kf_curve = self.plot.plot([], [], pen='b', name='KF')
        
        self.update_kalman()

    def create_slider(self, name, parent_layout, min_val, max_val, default_value):
        container = QHBoxLayout()
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_value)  # Midpoint

        label = QLabel(f"{name}: {slider.value()/self.var_scale:.8f}")  # MATCH slider value
        container.addWidget(label)
        container.addWidget(slider)
        parent_layout.addLayout(container)

        def on_slider_change(value, lbl=label, nm=name):
            lbl.setText(f"{nm}: {value/self.var_scale:.8f}")
            self.schedule_update_kalman()

        slider.valueChanged.connect(on_slider_change)
        return slider, label

    def schedule_update_kalman(self):
        self.update_timer.start()

    def update_kalman(self):
        var_process = self.q_slider.value() / self.var_scale
        var_radar = self.radar_slider.value() / self.var_scale
        var_lidar = self.lidar_slider.value() / self.var_scale
        use_radar = self.radar_checkbox.isChecked()
        use_lidar = self.lidar_checkbox.isChecked()

        dt = 1
        kf = KalmanFilter(dim_x=6, dim_z=4)
        block_F = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
        kf.F[0:3, 0:3] = block_F
        kf.F[3:6, 3:6] = block_F
        block_H = np.array([[1,0,0],[0,1,0]])
        kf.H[0:2,0:3] = block_H
        kf.H[2:4,3:6] = block_H
        kf.P *= 10
        kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=var_process, block_size=2)
        kf.x = np.array([[8],[0],[0],[0],[0],[0]])

        x_kf, y_kf = [kf.x[0,0]], [kf.x[3,0]]
        time_data = sorted(radar_data + lidar_data, key=lambda x: x[1])
        last_t = time_data[0][1]

        for sensor, time, z in time_data:
            dt = (time - last_t)/1e6
            if dt > 0:
                last_t = time
                kf.F[0:3, 0:3] = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
                kf.F[3:6, 3:6] = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
                kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=var_process, block_size=2)

            if sensor == 'L':
                if not use_lidar:
                    continue
                kf.R = np.eye(4) * var_lidar
            else:
                if not use_radar:
                    continue
                kf.R = np.eye(4) * var_radar

            kf.predict()
            kf.update(z)
            x_kf.append(kf.x[0,0])
            y_kf.append(kf.x[3,0])

        if use_radar:
            radar_pts = [(z[0,0], z[2,0]) for _,_,z in radar_data]
            self.radar_curve.setData(*zip(*radar_pts))
        else:
            self.radar_curve.setData([], [])

        if use_lidar:
            lidar_pts = [(z[0,0], z[2,0]) for _,_,z in lidar_data]
            self.lidar_curve.setData(*zip(*lidar_pts))
        else:
            self.lidar_curve.setData([], [])
        self.kf_curve.setData(x_kf, y_kf)
        self.plot.enableAutoRange()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = KalmanApp()
    win.show()
    sys.exit(app.exec())
