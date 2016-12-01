State Estimation Project Fall 2016
Jordan Marr
jordan.marr@robotics.utias.utoronto.ca

This project is designed to use a state estimation framework to estimate the rigid transformation between a LiDAR and an IMU.

The project makes use of numpy, Point Cloud Library and its python wrapper, and Lee Clement's pykitti library for working with the KITTI Vision Benchmark Suite data.

To run:
1) 'pip install pykitti'
2) replace the "raw.py" file in "(PATH TO PYTHON LIBRARIES)/python2.7/dist-packages/pykitti" with the version from this repository 
3) 'python demo_odometry.py'
