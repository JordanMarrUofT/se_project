"""Batch Linear Gaussian sensor offset estimation using point cloud registration
Module 1 - Estimation of solely the 2DOF planar transform (other 4 DOFs known a priori)"""
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
		first_scan_wframe = world_points_wframe #for testing/debugging
		#print(dataset.T_w_cam0[i])
	else:
		world_points_wframe = np.append(world_points_wframe, np.dot(dataset.T_w_cam0[i] , velo_points_cframe) , axis = 1)

#Display the first scan in the world frame if desired
"""
f = plt.figure()
ax = f.add_subplot(111, projection='3d')


ax.scatter(first_scan_wframe[0,:],
            first_scan_wframe[1,:],
            first_scan_wframe[2,:])

ax.set_title('First scan (subsampled, world frame) & cam0 trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""

#Get the measurements y of sensor offset based on point cloud registration
#y = np.zeros(2*len(dataset.frame_range))

roof_length = 2.7
disturbances = np.linspace(0,roof_length,10)
for i in range(len(dataset.frame_range)):
    for j in range(len(disturbances)):
	for k in range(len(disturbances)):
	    x = disturbances[j]
	    y = disturbances[k]
	    #Ground truth transformation
	    T = np.identity(4)
	    T[0,3] = x
	    T[1,3] = y
	    #raw_input('Press Enter.')

	    #subsample the velodyne scans (TODO: Make the sampled points different than those used for world cloud)
	    velo_range = range(0, dataset.velo[i].shape[0], 100)

	    #allocate the array for storing the scan points
	    velo_points = np.zeros((len(velo_range),4))

	    #store the scan in homogeneous coordinates [x y z 1]
	    velo_points[:,0:3] = dataset.velo[i][velo_range, 0:3]
	    velo_points[:,3] = np.ones((len(velo_range)))
	    #velo_points_tf is 4 x npoints
	    velo_points_tf = np.dot(T , velo_points.T)
	    ##########################################################################################
	    source_points = velo_points_tf[0:3,:].T
	    source_points = source_points.astype(np.float32)
	    pc_source = pcl.PointCloud()
	    pc_source.from_array(source_points)

	    target_points = velo_points[:,0:3]
	    target_points = target_points.astype(np.float32)
	    pc_target = pcl.PointCloud()
	    pc_target.from_array(target_points)

	    _, alignment, _, fitness = icp(pc_source, pc_target)
	    #y[2*i] = alignment[0,3]
	    #y[2*i + 1] = alignment[2,3]
	    print('Alignment fitness at time k = ' + str(i) + '(x=' + str(x) +',y=' + str(y) + '): ' + str(fitness))
	    print('\tTranslation : [' + str(alignment[0,3]) + ',' + str(alignment[1,3]) + ',' + str(alignment[2,3]) + ']')
	    #T_recovered = np.identity(4)
	    #T_recovered[0,3] = alignment[0,3]
	    #T_recovered[2,3] = alignment[2,3]
	    #velo_points_icp = np.dot( T_recovered , velo_points_tf )
	    if (i==0) and (x==0.0) and (y==0.0):
		xerror = np.array(alignment[0,3]+x)
		yerror = np.array(alignment[1,3]+y)
	    else:
		xerror = np.append(xerror,alignment[0,3]+x)
		yerror = np.append(yerror,alignment[1,3]+y)

print(np.var(xerror))
print(np.var(yerror))
