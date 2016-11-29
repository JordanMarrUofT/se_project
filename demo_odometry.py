"""Example of pykitti.odometry usage."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pykitti

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Change this to the directory where you store KITTI data
basedir = '/media/stars-jordan/Data/KITTI/odometry/dataset'

# Specify the dataset to load
sequence = '04'

# Optionally, specify the frame range to load
frame_range = range(0, 20, 5)

# Load the data
dataset = pykitti.odometry(basedir, sequence)
# dataset = pykitti.odometry(basedir, sequence, frame_range)

# Load some data
dataset.load_calib()        # Calibration data are accessible as named tuples
dataset.load_timestamps()   # Timestamps are parsed into timedelta objects
dataset.load_poses()        # Ground truth poses are loaded as 4x4 arrays
#dataset.load_gray()         # Left/right images are accessible as named tuples
#dataset.load_rgb()          # Left/right images are accessible as named tuples
dataset.load_velo()         # Each scan is a Nx4 array of [x,y,z,reflectance]

# Change this to the directory where you store KITTI data
basedir_raw = '/media/stars-jordan/Data/KITTI/raw'

# Specify the dataset to load
date = '2011_09_30'
drive = '0016'

# Optionally, specify the frame range to load
frame_range = range(0, 20, 5)

# Load the data
dataset_raw = pykitti.raw(basedir_raw, date, drive)
# dataset_raw = pykitti.raw(basedir_raw, date, drive, frame_range)

# Load some data
dataset_raw.load_calib()        # Calibration data are accessible as named tuples
dataset_raw.load_timestamps()   # Timestamps are parsed into datetime objects
dataset_raw.load_oxts()         # OXTS packets are loaded as named tuples
#dataset_raw.load_gray()         # Left/right images are accessible as named tuples
#dataset_raw.load_rgb()          # Left/right images are accessible as named tuples
dataset_raw.load_velo()         # Each scan is a Nx4 array of [x,y,z,reflectance]

# Display some of the data
np.set_printoptions(precision=4, suppress=True)
print('\nSequence: ' + str(dataset.sequence))
print('\nFrame range: ' + str(dataset.frame_range))

print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

print('\nFirst timestamp: ' + str(dataset.timestamps[0]))
print('\nSecond ground truth pose:\n' + str(dataset.T_w_cam0[1]))

''''
f, ax = plt.subplots(2, 2, figsize=(15, 5))
print(dataset.gray[0])
ax[0, 0].imshow(dataset.gray[0].left, cmap='gray')
ax[0, 0].set_title('Left Gray Image (cam0)')

ax[0, 1].imshow(dataset.gray[0].right, cmap='gray')
ax[0, 1].set_title('Right Gray Image (cam1)')

ax[1, 0].imshow(dataset.rgb[0].left)
ax[1, 0].set_title('Left RGB Image (cam2)')

ax[1, 1].imshow(dataset.rgb[0].right)
ax[1, 1].set_title('Right RGB Image (cam3)')
'''


# Plot every 100th point so things don't get too bogged down
velo_range = range(0, dataset.velo[2].shape[0], 100)
f2 = plt.figure()
ax2 = f2.add_subplot(111, projection='3d')
ax2.scatter(dataset.velo[2][velo_range, 0],
            dataset.velo[2][velo_range, 1],
            dataset.velo[2][velo_range, 2],
            c=dataset.velo[2][velo_range, 3],
            cmap='gray')
ax2.set_title('Third Velodyne scan (subsampled)')

f4 = plt.figure()
ax4 = f4.add_subplot(111, projection='3d')
ax4.scatter(dataset.velo[100][velo_range, 0],
            dataset.velo[100][velo_range, 1],
            dataset.velo[100][velo_range, 2],
            c=dataset.velo[100][velo_range, 3],
            cmap='gray')
ax4.set_title('101st Velodyne scan (subsampled)')
plt.xlabel('x')
plt.ylabel('y')
#plt.zlabel('z')

f3 = plt.figure()
ax3 = f3.add_subplot(111, projection='3d')

velo_points = np.zeros((1239,4))
velo_points[:,0:3] = dataset.velo[2][velo_range, 0:3]
velo_points[:,3] = np.ones((1239))
velo_points_cframe = np.dot(dataset_raw.calib.T_cam0unrect_velo , velo_points.T)

ax3.scatter(velo_points_cframe[0,:],
            velo_points_cframe[1,:],
            velo_points_cframe[2,:],
            c=dataset.velo[2][velo_range, 3],
            cmap='gray')
ax3.set_title('Third Velodyne scan (subsampled, cam0 frame)')
'''

'''


plt.show()
