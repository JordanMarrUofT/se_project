"""Batch Linear Gaussian sensor offset estimation using point cloud registration
Module 2 - estimating full 6DOF offset (not using the psedo-velo frame)"""

import pcl
from pcl.registration import icp, gicp, icp_nl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from util import pose_inv_map, rot_for_map, pose_for_map

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
frame_range = range(0, 100, 1)

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
print('\n100th OXTS packet:\n' + str(dataset_raw.oxts[99].packet))
print(dataset.timestamps[10])
print(dataset.timestamps[19].seconds + float(dataset.timestamps[19].microseconds)/1000000)

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
#invert the transformation
T_imu_cam0 = np.identity(4)
T_imu_cam0[0:3,0:3] = dataset_raw.calib.T_cam0_imu[0:3,0:3].T
T_imu_cam0[0:3,3] = -1 * np.dot( dataset_raw.calib.T_cam0_imu[0:3,0:3].T , dataset_raw.calib.T_cam0_imu[0:3,3] )


y = np.zeros(6*len(dataset.frame_range))
omega = np.zeros(6*len(dataset.frame_range))

K = len(dataset.frame_range) - 1 #number of timesteps (one less than number of indices because we have index for timestep 0)
x_check_f = np.zeros((8,4,K+1))
x_hat_f = np.zeros((8,4,K+1))
x_hat = np.zeros((8,4,K+1))
P_check_f = np.zeros((12,12,K+1))
P_hat_f = np.zeros((12,12,K+1))
P_hat = np.zeros((12,12,K+1))

T_v_w_meas = np.zeros((4,4,K+1))

x_check_f[0:4,0:4,0] = np.identity(4)
x_check_f[4:8,0:4,0] = np.identity(4)
x_check_f[4:7,0:3,0] = np.array([[0,0,1],[-1,0,0],[0,-1,0]])#alter the rotational component of the believed first pose to account for different axis definitions between the world and IMU frames (otherwise ICP won't converge) TODO: confirm this is right
P_check_f[0:6,0:6,0] = np.array([[0.816,0,0,0,0,0],[0,0.11,0,0,0,0],[0,0,0.816,0,0,0],[0,0,0,4.39,0,0],[0,0,0,0,4.39,0],[0,0,0,0,0,4.39]]) #calibration: 3-sigma = either 2.7m, 1m or 2pi rads
P_check_f[6:12,6:12,0] = np.array([[0.01,0,0,0,0,0],[0,0.01,0,0,0,0],[0,0,0.01,0,0,0],[0,0,0,0.000034,0,0],[0,0,0,0,0.000034,0],[0,0,0,0,0,0.000034]]) #position: 3-sigma = either 0.3m or pi/180 rads (1 degree)
R_k = np.array([[0.096,0,0,0,0,0],[0,0.0003,0,0,0,0],[0,0,0.15,0,0,0],[0,0,0,0.00005,0,0],[0,0,0,0,0.00005,0],[0,0,0,0,0,0.00005]])#from get_R_2.py

w_sigma = 0.000175 #rad/s (from OXTS user manual angular rate sigma value)
v_sigma = 0.0139 #m/s (from OXTS user manual velocity rms error (roughly corresponds to 1 sigma))
Q_k_rate = np.identity(6)
Q_k_rate[0:3,0:3] = np.square(v_sigma)*np.identity(3)
Q_k_rate[3:6,3:6] = np.square(w_sigma)*np.identity(3)
Q_k = np.zeros((12,12)) #preallocate the 12x12 matrix that will be used in EKF step 2

#preallocate G_k (doesn't change with time)
G_k = np.zeros((6,12))
G_k[0:6,0:6] = np.identity(6)
G_k[0:6,6:12] = np.identity(6)

for i in range(len(dataset.frame_range)):
	omega[6*i] = dataset_raw.oxts[i].packet.vf
	omega[6*i + 1] = dataset_raw.oxts[i].packet.vl
	omega[6*i + 2] = dataset_raw.oxts[i].packet.vu
	omega[6*i + 3] = dataset_raw.oxts[i].packet.wf
	omega[6*i + 4] = dataset_raw.oxts[i].packet.wl
	omega[6*i + 5] = dataset_raw.oxts[i].packet.wu	
	if i>0:
		delta_t = dataset.timestamps[i].seconds + float(dataset.timestamps[i].microseconds)/1000000
		delta_t = delta_t - (dataset.timestamps[i-1].seconds + float(dataset.timestamps[i-1].microseconds)/1000000)
		delta_p = delta_t*omega[6*i:(6*i + 6)].T #transpose to make it a column

		Q_k_IMU = np.square(delta_t)*Q_k_rate #convert rate uncertainty to pose change uncertainty
		
		#Pose change forward mapping
		a = delta_p[3:6]#column of rotational deltas
		phi = np.linalg.norm(a)
		a = (1/phi)*a
		a_skew = np.array([[0,-1*a[2],a[1]],[a[2],0,-1*a[0]],[-1*a[1],a[0],0]])

		C = np.cos(phi)*np.identity(3)
		C = C + (1-np.cos(phi))*np.dot( a , a.T )
		C = C + np.sin(phi)*a_skew

		J = (1/phi)*np.sin(phi)*np.identity(3)
		J = J + (1 - (1/phi)*np.sin(phi))*np.dot( a , a.T )
		J = J + (1/phi)*(1 - np.cos(phi))*a_skew

		r = np.dot( J , delta_p[0:3] )

		Xi = np.identity(4)
		Xi[0:3,0:3] = C
		Xi[0:3,3] = r

		state_tr_mat = np.identity(8)
		state_tr_mat[4:8,4:8] = Xi
		x_check_f[:,:,i] = np.dot(state_tr_mat , x_hat_f[:,:,(i-1)])

		Ad_Xi = np.zeros((6,6))
		Ad_Xi[0:3,0:3] = Xi[0:3,0:3]
		Ad_Xi[3:6,3:6] = Xi[0:3,0:3]
		r_skew = np.array([[0,-1*r[2],r[1]],[r[2],0,-1*r[0]],[-1*r[1],r[0],0]])
		Ad_Xi[0:3,3:6] = np.dot( r_skew , Xi[0:3,0:3] )#lec 9, slide 47
		cov_tr_mat = np.identity(12)
		cov_tr_mat[6:12,6:12] = Ad_Xi
		
		Q_k[6:12,6:12] = Q_k_IMU
		P_check_f[:,:,i] = np.dot(cov_tr_mat , np.dot(P_hat_f[:,:,(i-1)] , cov_tr_mat.T) ) + Q_k

	T_v_w = np.dot( x_check_f[0:4,:], x_check_f[4:8,:])#second term is world->imu, first is imu->velodyne
	world_points_vframe = np.dot( T_v_w , world_points_wframe )

	
	velo_range = range(0, dataset.velo[i].shape[0], 100)
	#allocate the array for storing the scan points
	velo_points = np.zeros((len(velo_range),4))

	#store the scan in homogeneous coordinates [x y z 1]
	velo_points[:,0:3] = dataset.velo[i][velo_range, 0:3]
	velo_points[:,3] = np.ones((len(velo_range)))


	source_points = world_points_vframe[0:3,:].T
	source_points = source_points.astype(np.float32)
	pc_source = pcl.PointCloud()
	pc_source.from_array(source_points)

	target_points = velo_points[:,0:3]
	target_points = target_points.astype(np.float32)
	pc_target = pcl.PointCloud()
	pc_target.from_array(target_points)

	_, T_v_w_resid, _, fitness = icp(pc_source, pc_target)

	T_v_w_meas[:,:,i] = np.dot(T_v_w_resid , T_v_w )
	

	print('Alignment fitness at time k = ' + str(i) + ': ' + str(fitness))
	print('\tworld->velo translation :\n\n' + str(T_v_w_meas[:,:,i]))

	kalman_denom = np.inv( np.dot( G_k , np.dot( P_check_f[:,:,i] , G_k.T) ) + R_k )
	kalman = np.dot( P_check_f[:,:,i] , np.dot(G_k.T , kalman_denom) ) #EKF step 3

	P_hat_f[:,:,i] = np.dot( (np.identity(12)-np.dot(kalman , G_k)) , P_check_f[:,:,i] )

	#get error term in Lie algebra (6x1 column)
	error = pose_inv_map(np.dot(T_v_w_meas , np.inv(np.dot( x_check_f[0:4,:], x_check_f[4:8,:]))))#compare measured (ICP) world->velo transform to what the state says it should be TODO: this might just be the same as T_v_w_resid so perhaps just need pose_inv_map(T_v_w_resid)

	eps_k = np.dot(kalman , error) #12x1
	state_hat_tr_mat = np.zeros((8,8))
	state_hat_tr_mat[0:4,0:4] = pose_for_map(eps_k[0:6])
	state_hat_tr_mat[4:8,4:8] = pose_for_map(eps_k[6:12])

	x_hat_f[:,:,i] = np.dot(state_hat_tr_mat , x_check_f[:,:,i])


