dATA taken from

https://github.com/jeremy-shannon/CarND-Extended-Kalman-Filter-Project/tree/master/data

Reference:
http://jeremyshannon.com/2017/04/21/udacity-sdcnd-extended-kalman-filter.html



Radar Measurement Format (R)



R   ρ   φ   ρ̇   timestamp   x_gt   y_gt   vx_gt   vy_gt
R	8.46642	0.0287602	-3.04035	1477010443399637	8.6	0.25	-3.00029	0

    R → Radar measurement identifier
    ρ (range) → Distance from the origin to the detected object (meters)
    φ (angle) → Bearing angle from the sensor to the object (radians)
    ρ̇ (range rate) → Radial velocity (speed of the object along the line of sight) (m/s)
    timestamp → Time of measurement (microseconds)
    x_gt (ground truth x) → True x-position of the object
    y_gt (ground truth y) → True y-position of the object
    vx_gt (ground truth vx) → True velocity in the x-direction
    vy_gt (ground truth vy) → True velocity in the y-direction

Lidar Measurement Format (L)

L   x   y   timestamp   x_gt   y_gt   vx_gt   vy_gt
L	8.44818	0.251553	1477010443449633	8.45	0.25	-3.00027	0

    L → Lidar measurement identifier
    x → Measured x-position of the object (meters)
    y → Measured y-position of the object (meters)
    timestamp → Time of measurement (microseconds)
    x_gt (ground truth x) → True x-position of the object
    y_gt (ground truth y) → True y-position of the object
    vx_gt (ground truth vx) → True velocity in the x-direction
    vy_gt (ground truth vy) → True velocity in the y-direction
