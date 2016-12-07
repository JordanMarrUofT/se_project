"""Batch Linear Gaussian sensor offset estimation using point cloud registration
Module 2 - estimating full 6DOF offset (not using the psedo-velo frame)"""

import pcl
from pcl.registration import icp, gicp, icp_nl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pykitti

# Change this to the directory where you store KITTI odometry data
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

# Change this to the directory where you store KITTI raw data
basedir_raw = '/media/stars-jordan/Data/KITTI/raw'

# Specify the dataset to load
date = '2011_09_30'
drive = '0020'

# Optionally, specify the frame range to load
#Raw dataset frame_range doesn't really matter for our purposes as we just need the velo->cam0 transform from it
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

np.set_printoptions(precision=4, suppress=True)

#allocate the trajectory array
traj = np.zeros((3,len(dataset.frame_range)))

#create the world-frame whole point cloud
for i in range(len(dataset.frame_range)):

	#subsample the velodyne scans
	velo_range = range(0, dataset.velo[i].shape[0], 100)

	#allocate the array for storing the scan points
	velo_points = np.zeros((len(velo_range),4))

	#store the scan in homogeneous coordinates [x y z 1]
	velo_points[:,0:3] = dataset.velo[i][velo_range, 0:3]
	velo_points[:,3] = np.ones((len(velo_range)))
	
	#KITTI doesn't provide the velo->world frame transform, but we can get this by an intermediate step (velo->cam0->world frame)
	velo_points_cframe = np.dot(dataset_raw.calib.T_cam0unrect_velo , velo_points.T)

	#grab the trajectory file
	traj[:,i] = -1*dataset.T_w_cam0[i][0:3,3]

	if i == 0:
		#world_points_wframe is 4 x npoints
		world_points_wframe = np.dot(dataset.T_w_cam0[i] , velo_points_cframe)
		first_scan_cframe = velo_points_cframe #for testing/debugging
	else:
		world_points_wframe = np.append(world_points_wframe, np.dot(dataset.T_w_cam0[i] , velo_points_cframe) , axis = 1)

#Display the world point cloud if desired
"""
f = plt.figure()
ax = f.add_subplot(111, projection='3d')


ax.scatter(world_points_wframe[0,:],
            world_points_wframe[1,:],
            world_points_wframe[2,:])

ax.scatter(traj[0,:],
            traj[1,:],
            traj[2,:],
	    color='r')
ax.set_title('Whole Velodyne cloud (subsampled, world frame) & cam0 trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""

#Get the measurements y of sensor offset based on point cloud registration
y = np.zeros(2*len(dataset.frame_range))
for i in range(len(dataset.frame_range)):
	#Build the world->cam0 transform from the cam0->world transform
	T_cam0_w = np.zeros((4,4))
	T_cam0_w[3,3] = 1
	T_cam0_w[0:3,0:3] = dataset.T_w_cam0[i][0:3,0:3].T
	T_cam0_w[0:3,3] = -1 * np.dot( dataset.T_w_cam0[i][0:3,0:3].T , dataset.T_w_cam0[i][0:3,3] )

	#the whole world point cloud, expressed in the camera's frame at the current instant
	world_points_cframe = np.dot( T_cam0_w , world_points_wframe )
	
	velo_range = range(0, dataset.velo[i].shape[0], 100)
	#allocate the array for storing the scan points
	velo_points = np.zeros((len(velo_range),4))

	#store the scan in homogeneous coordinates [x y z 1]
	velo_points[:,0:3] = dataset.velo[i][velo_range, 0:3]
	velo_points[:,3] = np.ones((len(velo_range)))

	###########try incorporating the trajectory to give better initial guess for alignment###############
	#velo_points = np.dot( T_cam0_w , velo_points.T )
	#TODO: Move this array creation outside the loop (it's not time dependent)
	#(p-v frame is the velodyne frame, but rotated to match the cam0 frame, and 
	# aligned along the normal
	# to the driving plane. This leaves only a planar transform between the psuedo-velo and
	# cam0 frames)
	T_pv_v = np.zeros((4,4))
	T_pv_v[0:3,0:3] = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
	T_pv_v[3,3] = 1

	velo_points = np.dot(T_pv_v, velo_points.T)
	#####################################################################################################

	source_points = velo_points[0:3,:].T
	source_points = source_points.astype(np.float32)
	pc_source = pcl.PointCloud()
	pc_source.from_array(source_points)

	target_points = world_points_cframe[0:3,:].T
	target_points = target_points.astype(np.float32)
	pc_target = pcl.PointCloud()
	pc_target.from_array(target_points)

	_, T_cam0_v, _, fitness = icp(pc_source, pc_target)
	#TODO: Construct the y values (possibly using Lie algebra)
	"""y[2*i] = alignment[0,3]
	y[2*i + 1] = alignment[2,3]"""

	print('Alignment fitness at time k = ' + str(i) + ': ' + str(fitness))
	print('\tvelo->cam0 translation :\n\n' + str(T_cam0_v))

	velo_points_cam0frame = np.dot( T_cam0_v , velo_points )

	###Sanity check
	
	if i == 0:
		f2 = plt.figure()
		ax2 = f2.add_subplot(111, projection='3d')


		ax2.scatter(velo_points_cam0frame[0,:],
            		velo_points_cam0frame[1,:],
            		velo_points_cam0frame[2,:],
	    		color='r')

		ax2.scatter(first_scan_cframe[0,:],
            		first_scan_cframe[1,:],
            		first_scan_cframe[2,:])
		ax2.set_title('Target and aligned-source velo scans at first instant, cam0 frame')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.show()
	
	###

