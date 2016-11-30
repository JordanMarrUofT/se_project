"""Example of pykitti.odometry usage."""
import pcl
from pcl.registration import icp, gicp, icp_nl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pykitti

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Change this to the directory where you store KITTI data
basedir = '/media/stars-jordan/Data/KITTI/odometry/dataset'

# Specify the dataset to load
sequence = '06'

# Optionally, specify the frame range to load
frame_range = range(0, 20, 1)

# Load the data
# dataset = pykitti.odometry(basedir, sequence)
dataset = pykitti.odometry(basedir, sequence, frame_range)

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
drive = '0020'

# Optionally, specify the frame range to load
frame_range = range(0, 20, 5)

# Load the data
# dataset_raw = pykitti.raw(basedir_raw, date, drive)
dataset_raw = pykitti.raw(basedir_raw, date, drive, frame_range)

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


traj = np.zeros((3,len(dataset.frame_range)))
for i in range(len(dataset.frame_range)):
	velo_range = range(0, dataset.velo[i].shape[0], 100)
	#print(len(velo_range))
	velo_points = np.zeros((len(velo_range),4))
	velo_points[:,0:3] = dataset.velo[i][velo_range, 0:3]
	velo_points[:,3] = np.ones((len(velo_range)))
	velo_points_cframe = np.dot(dataset_raw.calib.T_cam0unrect_velo , velo_points.T)

	traj[:,i] = -1*dataset.T_w_cam0[i][0:3,3]


	if i == 0:
		velo_points_wframe = np.dot(dataset.T_w_cam0[i] , velo_points_cframe)
		#velo_points_wframe = velo_points_cframe
	elif i == 2:
		second_scan_wframe = np.dot(dataset.T_w_cam0[i] , velo_points_cframe)
		
		velo_points_wframe = np.append(velo_points_wframe, second_scan_wframe , axis = 1)
	else:
		velo_points_wframe = np.append(velo_points_wframe, np.dot(dataset.T_w_cam0[i] , velo_points_cframe) , axis = 1)
		#velo_points_wframe = np.append(velo_points_wframe, velo_points_cframe , axis = 1)

print(traj.T)
f = plt.figure()
ax = f.add_subplot(111, projection='3d')


ax.scatter(velo_points_wframe[0,:],
            velo_points_wframe[1,:],
            velo_points_wframe[2,:])

ax.scatter(traj[0,:],
            traj[1,:],
            traj[2,:],
	    color='r')
ax.set_title('Whole Velodyne cloud (subsampled, world frame)')
plt.xlabel('x')
plt.ylabel('y')
	

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
plt.xlabel('x')
plt.ylabel('y')
"""
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
"""

velo_range = range(0, dataset_raw.velo[2].shape[0], 100)
f3 = plt.figure()
ax3 = f3.add_subplot(111, projection='3d')

velo_points = np.zeros((len(velo_range),4))
velo_points[:,0:3] = dataset.velo[2][velo_range, 0:3]
velo_points[:,3] = np.ones((len(velo_range)))
velo_points_cframe = np.dot(dataset_raw.calib.T_cam0unrect_velo , velo_points.T)

ax3.scatter(velo_points_cframe[0,:],
            velo_points_cframe[1,:],
            velo_points_cframe[2,:],
            c=dataset.velo[2][velo_range, 3],
            cmap='gray')
ax3.set_title('Third Velodyne scan (subsampled, cam0 frame)')
"""
ax3.scatter(traj[0,2],
            traj[1,2],
            traj[2,2],
	    color='r')
"""
ax3.scatter(-dataset_raw.calib.T_cam0unrect_velo[0,3],
            -dataset_raw.calib.T_cam0unrect_velo[1,3],
            -dataset_raw.calib.T_cam0unrect_velo[2,3],
	    color='r')


plt.show()

source_points = velo_points_cframe[0:3,:].T
source_points = source_points.astype(np.float32)
pc_source = pcl.PointCloud()
pc_source.from_array(source_points)

target_points = velo_points_wframe[0:3,:].T
target_points = target_points.astype(np.float32)
pc_target = pcl.PointCloud()
pc_target.from_array(target_points)

_, alignment, _, _ = icp(pc_source, pc_target)


f5 = plt.figure()
ax5 = f5.add_subplot(111, projection='3d')

aligned_source = np.dot(alignment, velo_points_cframe)

ax5.scatter(aligned_source[0,:],
            aligned_source[1,:],
            aligned_source[2,:],
            color='r')
#ax3.set_title('Third Velodyne scan (subsampled, cam0 frame)')

ax5.scatter(second_scan_wframe[0,:],
            second_scan_wframe[1,:],
            second_scan_wframe[2,:])


plt.show()
