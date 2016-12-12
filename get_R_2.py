"""Based on "Robot Localization Based on Scan-matching - estimating the covariance matrix for the IDC algorithm" By Bengtsson and Baerveldt (2003)"""
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

roof_length = 2.7 #m
rot_stddev_degrees = 0.3 #degrees (1/3 of the "3-sigma" range)
rot_stddev = rot_stddev_degrees*np.pi/180 #radians
stddev = np.array([np.sqrt(roof_length/3), np.sqrt(roof_length/3), np.sqrt(roof_length/3), rot_stddev, rot_stddev, rot_stddev])

x_error = np.zeros(100)
y_error = np.zeros(100)
z_error = np.zeros(100)
th1_error = np.zeros(100)
th2_error = np.zeros(100)
th3_error = np.zeros(100)
fitness_arr = np.zeros(100)
for i in range(100):
	x_dist = np.random.normal(loc=0.0, scale=stddev[0])#dist as in disturbance
	y_dist = np.random.normal(loc=0.0, scale=stddev[1])
	z_dist = np.random.normal(loc=0.0, scale=stddev[2])
	th1_dist = np.random.normal(loc=0.0, scale=stddev[3])
	th2_dist = np.random.normal(loc=0.0, scale=stddev[4])
	th3_dist = np.random.normal(loc=0.0, scale=stddev[5])
	
	a = np.array([[th1_dist],[th2_dist],[th3_dist]])#column
	phi = np.linalg.norm(a)
	a = (1/phi)*a
	a_skew = np.array([[0,-1*a[2],a[1]],[a[2],0,-1*a[0]],[-1*a[1],a[0],0]])

	C_dist = np.cos(phi)*np.identity(3)
	C_dist = C_dist + (1-np.cos(phi))*np.dot( a , a.T )
	C_dist = C_dist + np.sin(phi)*a_skew


	T_dist = np.identity(4)
	T_dist[0:3,0:3] = C_dist
	T_dist[0,3] = x_dist
	T_dist[1,3] = y_dist
	T_dist[2,3] = z_dist

	
	

	first_scan_dist = np.dot( T_dist , first_scan_wframe )

	source_points = first_scan_dist[0:3,:].T
	source_points = source_points.astype(np.float32)
	pc_source = pcl.PointCloud()
	pc_source.from_array(source_points)

	target_points = world_points_wframe[0:3,:].T
	target_points = target_points.astype(np.float32)
	pc_target = pcl.PointCloud()
	pc_target.from_array(target_points)

	_, T_recovered_inv, _, fitness = icp(pc_source, pc_target)#this is the inverse transformation to the one we applied
	fitness_arr[i] = fitness


	#Build the transformation from its inverse
	T_recovered = np.identity(4)
	T_recovered[0:3,0:3] = T_recovered_inv[0:3,0:3].T
	T_recovered[0:3,3] = -1 * np.dot( T_recovered_inv[0:3,0:3].T , T_recovered_inv[0:3,3] )


	w, v = np.linalg.eig(T_recovered[0:3,0:3])
	

	index = np.argmin(np.absolute(w.imag))
	#print(w[index])
	a_recovered = v[:,index].real
	#print(a_recovered)

	phi_recovered = np.arccos( (0.5)*(np.trace(T_recovered[0:3,0:3])-1) )

	x_error[i] = T_recovered[0,3] - x_dist
	y_error[i] = T_recovered[1,3] - y_dist
	z_error[i] = T_recovered[2,3] - z_dist
	th1_error[i] = phi_recovered*a_recovered[0] - th1_dist
	th2_error[i] = phi_recovered*a_recovered[1] - th2_dist
	th3_error[i] = phi_recovered*a_recovered[2] - th3_dist

print('variance in x: ' + str(np.var(x_error)))
print('variance in y: ' + str(np.var(y_error)))
print('variance in z: ' + str(np.var(z_error)))
print('variance in theta_1: ' + str(np.var(th1_error)))
print('variance in theta_2: ' + str(np.var(th2_error)))
print('variance in theta_3: ' + str(np.var(th3_error)))

