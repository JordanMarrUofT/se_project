"""Batch Linear Gaussian sensor offset estimation using point cloud registration
Module 3 - estimating full 6DOF offset & imu pose"""

import pcl
from pcl.registration import icp, gicp, icp_nl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from util import pose_inv_map, rot_for_map, pose_for_map, invert_transform, skew, get_adjoint

import pykitti

# Change this to the directory where you store KITTI odometry data
basedir = '/media/stars-jordan/Data/KITTI/odometry/dataset'

# Specify the dataset to load
sequence = '06'

# Optionally, specify the frame range to load
frame_range = range(0, 50, 1)
total_num_batch_it = 100

# Load the data
# dataset = pykitti.odometry(basedir, sequence)
dataset = pykitti.odometry(basedir, sequence, frame_range)

# Load some data
dataset.load_calib()        # Calibration data are accessible as named tuples
dataset.load_timestamps()   # Timestamps are parsed into timedelta objects
dataset.load_poses()        # Ground truth poses are loaded as 4x4 arrays
dataset.load_velo()         # Each scan is a Nx4 array of [x,y,z,reflectance]

# Change this to the directory where you store KITTI raw data
basedir_raw = '/media/stars-jordan/Data/KITTI/raw'

# Specify the dataset to load
date = '2011_09_30'
drive = '0020'

# Optionally, specify the frame range to load
#Raw dataset frame_range doesn't really matter for our purposes as we just need the calibrations from it
frame_range = range(0, 100, 1)

# Load the data
# dataset_raw = pykitti.raw(basedir_raw, date, drive)
dataset_raw = pykitti.raw(basedir_raw, date, drive, frame_range)

# Load some data
dataset_raw.load_calib()        # Calibration data are accessible as named tuples
dataset_raw.load_timestamps()   # Timestamps are parsed into datetime objects
dataset_raw.load_oxts()         # OXTS packets are loaded as named tuples
dataset_raw.load_velo()         # Each scan is a Nx4 array of [x,y,z,reflectance]


np.set_printoptions(precision=4, suppress=True)

#allocate the trajectory array
traj_velo = np.zeros((3,len(dataset.frame_range)))
traj_imu = np.zeros((3,len(dataset.frame_range)))
traj_f = np.zeros((3,len(dataset.frame_range)))

#create the world-frame whole point cloud
for i in range(len(dataset.frame_range)):

	#subsample the velodyne scans
	velo_range = range(50, dataset.velo[i].shape[0], 100)#TODO: This line controls how many common points the source and target point clouds will have

	#allocate the array for storing the scan points
	velo_points = np.zeros((len(velo_range),4))

	#store the scan in homogeneous coordinates [x y z 1]
	velo_points[:,0:3] = dataset.velo[i][velo_range, 0:3]
	velo_points[:,3] = np.ones((len(velo_range)))
	
	#KITTI doesn't provide the velo->world frame transform, but we can get this by an intermediate step (velo->cam0->world frame)
	velo_points_cframe = np.dot(dataset_raw.calib.T_cam0unrect_velo , velo_points.T)

	#grab the trajectory file
	T_v_w_gt = invert_transform(np.dot(dataset.T_w_cam0[i], dataset_raw.calib.T_cam0unrect_velo))
	traj_velo[:,i] = -1*np.dot(T_v_w_gt[0:3,0:3].T , T_v_w_gt[0:3,3])
	T_i_w_gt = invert_transform(np.dot(dataset.T_w_cam0[i], dataset_raw.calib.T_cam0_imu))
	traj_imu[:,i] = -1*np.dot(T_i_w_gt[0:3,0:3].T , T_i_w_gt[0:3,3])

	if i == 0:
		#world_points_wframe is 4 x npoints
		world_points_wframe = np.dot(dataset.T_w_cam0[i] , velo_points_cframe)
	else:
		world_points_wframe = np.append(world_points_wframe, np.dot(dataset.T_w_cam0[i] , velo_points_cframe) , axis = 1)
		if i == 19:
			twentieth_scan_wframe = np.dot(dataset.T_w_cam0[i] , velo_points_cframe) #for testing/debugging

y = np.zeros(6*len(dataset.frame_range))
omega = np.zeros(6*len(dataset.frame_range))

K = len(dataset.frame_range) - 1 #number of timesteps (one less than number of indices because we have index for timestep 0)
x_check_f = np.zeros((8,4,K+1))
x_hat_f = np.zeros((8,4,K+1))
x_hat = np.zeros((8,4,K+1))
P_check_f = np.zeros((12,12,K+1))
P_hat_f = np.zeros((12,12,K+1))
P_hat = np.zeros((12,12,K+1))


#x_check_f[0:4,0:4,0] = dataset_raw.calib.T_velo_imu #TODO
x_check_f[0:4,0:4,0] = np.identity(4)
x_check_f[4:8,0:4,0] = invert_transform(np.dot(dataset.T_w_cam0[0], dataset_raw.calib.T_cam0_imu))
#x_check_f[4:7,3:4,0] = np.array([[1.2],[-0.3],[0.8]])#give good initial guess for IMU position so ICP converges
#x_check_f[4:7,0:3,0] = np.array([[0,0,1],[-1,0,0],[0,-1,0]])#alter the rotational component of the believed first pose to account for different axis definitions between the world and IMU frames (otherwise ICP won't converge) TODO: confirm this is right


P_check_f[0:6,0:6,0] = np.array([[0.816,0,0,0,0,0],[0,0.816,0,0,0,0],[0,0,0.11,0,0,0],[0,0,0,4.39,0,0],[0,0,0,0,4.39,0],[0,0,0,0,0,4.39]]) #calibration: 3-sigma = either 2.7m, 1m or 2pi rads

P_check_f[6:12,6:12,0] = np.array([[0.01,0,0,0,0,0],[0,0.01,0,0,0,0],[0,0,0.01,0,0,0],[0,0,0,0.000034,0,0],[0,0,0,0,0.000034,0],[0,0,0,0,0,0.000034]]) #position: 3-sigma = either 0.3m or pi/180 rads (1 degree)

R_k = np.array([[0.096,0,0,0,0,0],[0,0.0003,0,0,0,0],[0,0,0.15,0,0,0],[0,0,0,0.00005,0,0],[0,0,0,0,0.00005,0],[0,0,0,0,0,0.00005]])#from get_R_2.py

w_sigma = 0.000175 #rad/s (from OXTS user manual angular rate sigma value)
v_sigma = 0.0139 #m/s (from OXTS user manual velocity rms error (roughly corresponds to 1 sigma))
Q_k_rate = np.identity(6)
Q_k_rate[0:3,0:3] = np.square(v_sigma)*np.identity(3)
Q_k_rate[3:6,3:6] = np.square(w_sigma)*np.identity(3)
Q_k_rate = Q_k_rate

Q_k = np.zeros((12,12)) #preallocate the 12x12 matrix that will be used in EKF step 2

#preallocate G_k (doesn't change with time)
G_k = np.zeros((6,12))
G_k[0:6,0:6] = np.identity(6)
G_k[0:6,6:12] = np.identity(6)

W = np.identity(6*2*(K+1))#need this for batch section, but we can populate it in the ekf section to avoid calculating it twice
W[0:6,0:6] = P_check_f[6:12,6:12,0]
W[6*(K+1):6*(K+2),6*(K+1):6*(K+2)] = R_k

F = np.identity(6*(K+1))#same thing as W, need for batch but can populate in ekf

Xi = np.zeros((4,4,K+1)) #Xi[:,:,0] will just stay as zeros (don't define Xi_0 <- movement before first timestep)

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
		delta_p = -1*delta_p

		Q_k_IMU = np.square(delta_t)*Q_k_rate #convert rate uncertainty to pose change uncertainty
		Q_k[6:12,6:12] = Q_k_IMU

		W[6*i:(6*i+6),6*i:(6*i+6)] = Q_k_IMU
		W[6*(K+1+i):6*(K+2+i),6*(K+1+i):6*(K+2+i)] = R_k
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

		Xi[:,:,i] = np.identity(4)
		Xi[0:3,0:3,i] = C
		Xi[0:3,3,i] = r

		state_check_tr_mat = np.identity(8)
		state_check_tr_mat[4:8,4:8] = Xi[:,:,i]
		x_check_f[:,:,i] = np.dot(state_check_tr_mat , x_hat_f[:,:,(i-1)])

		Ad_Xi = np.zeros((6,6))
		
		Ad_Xi[0:3,0:3] = Xi[0:3,0:3,i]
		Ad_Xi[3:6,3:6] = Xi[0:3,0:3,i]
		r_skew = np.array([[0,-1*r[2],r[1]],[r[2],0,-1*r[0]],[-1*r[1],r[0],0]])
		Ad_Xi[0:3,3:6] = np.dot( r_skew , Xi[0:3,0:3,i] )#lec 9, slide 47
		"""
		Ad_Xi[0:3,0:3] = skew(delta_p[3:6])
		Ad_Xi[3:6,3:6] = Ad_Xi[0:3,0:3]
		Ad_Xi[0:3,3:6] = skew(delta_p[0:3])
		"""
		cov_tr_mat = np.identity(12)
		cov_tr_mat[6:12,6:12] = Ad_Xi

		F[6*i:6*(i+1),6*(i-1):6*i] = -1*Ad_Xi

		P_check_f[:,:,i] = np.dot(cov_tr_mat , np.dot(P_hat_f[:,:,(i-1)] , cov_tr_mat.T) ) + Q_k

	T_v_w = np.dot( x_check_f[0:4,:,i], x_check_f[4:8,:,i])#second term is world->imu, first is imu->velodyne	
	world_points_vframe = np.dot( T_v_w , world_points_wframe )

	
	velo_range = range(0, dataset.velo[i].shape[0], 100)
	#allocate the array for storing the scan points
	velo_points = np.zeros((len(velo_range),4))

	#store the scan in homogeneous coordinates [x y z 1]
	velo_points[:,0:3] = dataset.velo[i][velo_range, 0:3]
	velo_points[:,3] = np.ones((len(velo_range)))


	source_points = velo_points[:,0:3]
	source_points = source_points.astype(np.float32)
	pc_source = pcl.PointCloud()
	pc_source.from_array(source_points)

	target_points = world_points_vframe[0:3,:].T
	target_points = target_points.astype(np.float32)
	pc_target = pcl.PointCloud()
	pc_target.from_array(target_points)

	_, T_v_w_resid_inv, _, fitness = icp(pc_source, pc_target)
	T_v_w_resid = invert_transform(T_v_w_resid_inv)

	T_v_w_icp = np.dot(T_v_w_resid , T_v_w )#full world->velo transformation after applying icp correction
	

	#print('Alignment fitness at time k = ' + str(i) + ': ' + str(fitness))
	#print('\tworld->velo residual translation :\n\n' + str(T_v_w_resid))
	
	if i == 19:
		f2 = plt.figure()
		ax2 = f2.add_subplot(111, projection='3d')


		#ax2.scatter(velo_points[:,0],
            	#	velo_points[:,1],
            	#	velo_points[:,2],
	    	#	color='r')
		twentieth_scan_vframe = np.dot( T_v_w , twentieth_scan_wframe )
		twentieth_scan_vframe_post_icp = np.dot(T_v_w_icp , twentieth_scan_wframe)

		source_points = velo_points[:,0:3]
		source_points = source_points.astype(np.float32)
		pc_source = pcl.PointCloud()
		pc_source.from_array(source_points)

		target_points = twentieth_scan_vframe[0:3,:].T
		target_points = target_points.astype(np.float32)
		pc_target = pcl.PointCloud()
		pc_target.from_array(target_points)

		_, T_v_w_resid_inv_ss, _, fitness = icp(pc_source, pc_target)

		#print('Alignment fitness (scan-to-scan) at time k = ' + str(i) + ': ' + str(fitness))
		T_v_w_resid_ss = invert_transform(T_v_w_resid_inv_ss)

		T_v_w_meas_ss = np.dot(T_v_w_resid_ss , T_v_w )

		twentieth_scan_vframe_post_icp_ss = np.dot(T_v_w_meas_ss , twentieth_scan_wframe)

		ax2.scatter(twentieth_scan_vframe_post_icp[0,:],
            		twentieth_scan_vframe_post_icp[1,:],
            		twentieth_scan_vframe_post_icp[2,:])
		ax2.scatter(velo_points[:,0],
            		velo_points[:,1],
            		velo_points[:,2],
	    		color='r')
		ax2.set_title('Target and aligned-source velo scans at 20th instant, velo frame')
		plt.xlabel('x (m)')
		plt.ylabel('y (m)')
		plt.show()
	
	kalman_denom = np.linalg.inv( np.dot( G_k , np.dot( P_check_f[:,:,i] , G_k.T) ) + R_k )
	kalman = np.dot( P_check_f[:,:,i] , np.dot(G_k.T , kalman_denom) ) #EKF step 3

	P_hat_f[:,:,i] = np.dot( (np.identity(12)-np.dot(kalman , G_k)) , P_check_f[:,:,i] )

	#get error term in Lie algebra (6x1 column)
	T_v_w_state_inv = invert_transform(np.dot( x_check_f[0:4,:,i], x_check_f[4:8,:,i]))
	error = pose_inv_map(np.dot(T_v_w_icp , T_v_w_state_inv))#compare measured (ICP) world->velo transform to what the state says it should be TODO: this might just be the same as T_v_w_resid so perhaps just need pose_inv_map(T_v_w_resid)

	eps_k = np.dot(kalman , error) #12x1
	state_hat_tr_mat = np.zeros((8,8))
	state_hat_tr_mat[0:4,0:4] = pose_for_map(eps_k[0:6])#TODO
	#state_hat_tr_mat[0:4,0:4] = np.identity(4)
	state_hat_tr_mat[4:8,4:8] = pose_for_map(eps_k[6:12])

	x_hat_f[:,:,i] = np.dot(state_hat_tr_mat , x_check_f[:,:,i])
	traj_f[:,i] = x_hat_f[4:7,3,i]

e_pose_oneit = np.zeros((6,K+1))
e_pose_rms = np.zeros((6,total_num_batch_it+1))
e_calib = np.zeros((6,total_num_batch_it+1))

#seed x_op with the result from the EKF pass
for i in range(len(dataset.frame_range)):
	if i == 0:
		T_zero_check = x_hat_f[4:8,:,i] #will use this as a seed value, will show up in all calculations for e_v_0(x_op)
		x_op = T_zero_check
	else:
		x_op = np.append(x_op, x_hat_f[4:8,:,i] , axis = 0)
	T_i_w_gt = invert_transform(np.dot(dataset.T_w_cam0[i], dataset_raw.calib.T_cam0_imu))
	e_pose_oneit[:,i:i+1] = pose_inv_map(np.dot(T_i_w_gt , invert_transform(x_hat_f[4:8,:,i])))

x_op = np.append(x_op, x_hat_f[0:4,:,-1], axis = 0)#calib value from final timestep (likely the most accurate one)
#x_op = np.append(x_op, dataset_raw.calib.T_velo_imu , axis = 0)

e_calib[:,0:1] = pose_inv_map(np.dot(dataset_raw.calib.T_velo_imu , invert_transform(x_op[-4:,:])))
e_pose_rms[:,0] = np.sqrt(K+1)*np.linalg.norm(e_pose_oneit, axis=1)

traj_velo_b = np.zeros((3,len(dataset.frame_range)))
traj_imu_b = np.zeros((3,len(dataset.frame_range)))
traj_velo_meas = np.zeros((3,len(dataset.frame_range)))
traj_imu_meas = np.zeros((3,len(dataset.frame_range)))
T_v_w_meas=np.zeros((4,4,K+1))
e_y = np.zeros((6*(K+1),1))
e_v = np.zeros((6*(K+1),1))
#e_v[0:6,0:1] = ... will just be zeros on first pass (first x_op value is still just the T_zero_check value we seed it with)

fitness_vec = np.zeros((K+1))

for num_batch_it in range(total_num_batch_it):

	#get y values for the batch method
	for i in range(len(dataset.frame_range)):
		T_v_w = np.dot( x_op[-4:,:], x_op[(4*i):(4*i+4),:])#second term is world->imu, first is imu->velodyne	
		if num_batch_it == 0: 
		#only do the icp alignments once for the batch method (two times total because we used icp in ekf), this should work because once we have a "good enough" initial alignment from the ekf, the measured T_v_w shouldn't change, and more importantly recalculating it on each batch iteration could be problematic because of the sensitivity of icp when aligning to a large target
			world_points_vframe = np.dot( T_v_w , world_points_wframe )

	
			velo_range = range(0, dataset.velo[i].shape[0], 100)
			#allocate the array for storing the scan points
			velo_points = np.zeros((len(velo_range),4))

			#store the scan in homogeneous coordinates [x y z 1]
			velo_points[:,0:3] = dataset.velo[i][velo_range, 0:3]
			velo_points[:,3] = np.ones((len(velo_range)))


			source_points = velo_points[:,0:3]
			source_points = source_points.astype(np.float32)
			pc_source = pcl.PointCloud()
			pc_source.from_array(source_points)

			target_points = world_points_vframe[0:3,:].T
			target_points = target_points.astype(np.float32)
			pc_target = pcl.PointCloud()
			pc_target.from_array(target_points)

			_, T_v_w_resid_inv, _, fitness = icp(pc_source, pc_target)
			T_v_w_resid = invert_transform(T_v_w_resid_inv)
			fitness_vec[i] = fitness
		
			T_v_w_meas[:,:,i] = np.dot(T_v_w_resid , T_v_w )
			traj_velo_meas[:,i] = -1*np.dot(T_v_w_meas[0:3,0:3,i].T, T_v_w_meas[0:3,3,i])
			T_i_w_meas = np.dot(invert_transform(x_op[-4:,:]) , T_v_w_meas[:,:,i] )
			traj_imu_meas[:,i] = -1*np.dot(T_i_w_meas[0:3,0:3].T, T_i_w_meas[0:3,3])
	
		#get error term in Lie algebra (6x1 column)
		T_v_w_state_inv = invert_transform(T_v_w)
		e_y[6*i:(6*i+6),0:1] = pose_inv_map(np.dot(T_v_w_meas[:,:,i] , T_v_w_state_inv))#compare measured (ICP) world->velo transform to what the state says it should be TODO: this might just be the same as T_v_w_resid so perhaps just need pose_inv_map(T_v_w_resid)
		if i>0 and num_batch_it > 0:
			debugger = np.dot(Xi[:,:,i] , np.dot( x_op[4*(i-1):(4*(i-1)+4),:] , invert_transform(x_op[4*i:(4*i+4),:])))
			e_v[6*i:(6*i+6),0:1] = pose_inv_map(debugger)
			F[6*i:6*(i+1),6*(i-1):6*i] = -1*get_adjoint(np.dot(x_op[4*i:(4*i+4),:],invert_transform(x_op[4*(i-1):(4*(i-1)+4),:])))
	
	if num_batch_it > 0:
		e_v[0:6] = pose_inv_map(np.dot(T_zero_check , invert_transform(x_op[0:4,:])))

	e = np.append(e_v , e_y , axis = 0)
	e = np.nan_to_num(e)
	
	
	#create G
	G = np.identity(6*(K+1))
	ones_col = np.identity(6)
	for i in range(len(dataset.frame_range)):
		if i>0:
			ones_col = np.append(ones_col, np.identity(6), axis = 0)

	G = np.append(G, ones_col, axis=1)#horizontal append

	#finish creating F by appending zeros
	F = np.append(F[0:6*(K+1),0:6*(K+1)], np.zeros((6*(K+1),6)), axis=1)#horizontal append

	H = np.append(F, G, axis=0)#vertical append
	A = np.dot( H.T, np.linalg.solve(W, H))
	b = np.dot( H.T, np.linalg.solve(W, e))

	

	A_11 = A[0:6*(K+1),0:6*(K+1)]
	A_12 = A[0:6*(K+1),6*(K+1):6*(K+2)]
	A_22 = A[6*(K+1):6*(K+2),6*(K+1):6*(K+2)]

	L_11 = np.linalg.cholesky(A_11)
	L_21 = np.linalg.solve(L_11, A_12).T
	L_22 = np.linalg.cholesky(A_22 - np.dot(L_21, L_21.T))

	L_right = np.append(0*L_21.T, L_22, axis=0)# L_12 (which is just 0 of size equal to L_21 transpose), and L_22
	L = np.append(L_11, L_21, axis=0)
	L = np.append(L, L_right, axis=1)

	y = np.linalg.solve(L,b)
	dx = np.linalg.solve(L.T,y)


	for i in range(len(dataset.frame_range)):
		#Might need to insert adjustment here for dx=[0 0 0 0 0 0]

		x_op[4*i:4*(i+1),:] = np.dot(pose_for_map(dx[6*i:6*(i+1)]) , x_op[4*i:4*(i+1),:])#adjust imu poses
		T_v_w_b = np.dot(x_op[-4:,:], x_op[4*i:4*(i+1),:])
		traj_velo_b[:,i] = -1*np.dot(T_v_w_b[0:3,0:3].T , T_v_w_b[0:3,3])
		traj_imu_b[:,i] = -1*np.dot(x_op[4*i:4*(i+1)-1,0:3].T, x_op[4*i:4*(i+1)-1,3])
		T_i_w_gt = invert_transform(np.dot(dataset.T_w_cam0[i], dataset_raw.calib.T_cam0_imu))
		e_pose_oneit[:,i:i+1] = pose_inv_map(np.dot(T_i_w_gt , invert_transform(x_op[4*i:4*(i+1),:])))
			
	x_op[-4:,:] = np.dot(pose_for_map(dx[-6:]) , x_op[-4:,:])#adjust calibration

	e_calib[:,num_batch_it+1:num_batch_it+2] = pose_inv_map(np.dot(dataset_raw.calib.T_velo_imu , invert_transform(x_op[-4:,:])))
	e_pose_rms[:,num_batch_it+1] = np.sqrt(K+1)*np.linalg.norm(e_pose_oneit, axis=1)

	#print(pose_for_map(dx[-6:]))
	#print('Calibration (T_v_i):')
	#print(x_op[-4:,:])
	"""
	f2 = plt.figure()
	ax2 = f2.add_subplot(111, projection='3d')


	ax2.scatter(traj_velo_b[0,:],
            traj_velo_b[1,:],
            traj_velo_b[2,:])

	ax2.scatter(traj_velo[0,:],
            traj_velo[1,:],
            traj_velo[2,:],
	    color='r')

	ax2.scatter(traj_velo_meas[0,:],
            traj_velo_meas[1,:],
            traj_velo_meas[2,:],
	    color='g')

	ax2.set_title('velo estimated (blue), measured (green) & actual (red) trajectory, post batch iteration #' + str(num_batch_it))
	plt.xlabel('x')
	plt.ylabel('y')
	plt.xlim((-30,0))
	plt.ylim((-30,0))

	f3 = plt.figure()
	ax3 = f3.add_subplot(111, projection='3d')


	ax3.scatter(traj_imu_b[0,:],
            traj_imu_b[1,:],
            traj_imu_b[2,:])
	
	ax3.scatter(traj_imu_meas[0,:],
            traj_imu_meas[1,:],
            traj_imu_meas[2,:],
	    color='g')

	ax3.scatter(traj_imu[0,:],
            traj_imu[1,:],
            traj_imu[2,:],
	    color='r')

	ax3.set_title('imu estimated (blue) & actual (red) trajectory, post batch iteration #' + str(num_batch_it))
	plt.xlabel('x')
	plt.ylabel('y')
	plt.xlim((-30,0))
	plt.ylim((-30,0))

	plt.show()
	"""

f2 = plt.figure()
ax2 = f2.add_subplot(111, projection='3d')


ax2.scatter(traj_velo_b[0,:],
            traj_velo_b[1,:],
            traj_velo_b[2,:])

ax2.scatter(traj_velo[0,:],
            traj_velo[1,:],
            traj_velo[2,:],
	    color='r')

ax2.scatter(traj_velo_meas[0,:],
            traj_velo_meas[1,:],
            traj_velo_meas[2,:],
	    color='g')

ax2.set_title('velo estimated (blue), measured (green) & actual (red) trajectory, post batch iteration #' + str(num_batch_it+1))
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.xlim((-3,0))
plt.ylim((-3,0))

f3 = plt.figure()
ax3 = f3.add_subplot(111, projection='3d')


ax3.scatter(traj_imu_b[0,:],
            traj_imu_b[1,:],
            traj_imu_b[2,:])
	
ax3.scatter(traj_imu_meas[0,:],
            traj_imu_meas[1,:],
            traj_imu_meas[2,:],
	    color='g')

ax3.scatter(traj_imu[0,:],
            traj_imu[1,:],
            traj_imu[2,:],
	    color='r')

ax3.set_title('imu estimated (blue), measured (green) & actual (red) trajectory, post batch iteration #' + str(num_batch_it+1))
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.xlim((-3,0))
plt.ylim((-3,0))

plt.show()

e_calib_t = np.linalg.norm(e_calib[0:3,:], axis=0)
e_calib_r = np.linalg.norm(e_calib[3:6,:], axis=0)

e_pose_rms_t = np.mean(e_pose_rms[0:3,:], axis=0)
e_pose_rms_r = np.mean(e_pose_rms[3:6,:], axis=0)

plt.figure()
plt.plot(e_calib_t[1:])
plt.xlabel('Batch solver iterations')
plt.ylabel('Calibration translational error (m)')

plt.figure()
plt.plot(e_calib_r[1:])
plt.xlabel('Batch solver iterations')
plt.ylabel('Calibration rotational error (rads)')

plt.figure()
plt.plot(e_pose_rms_t[1:])
plt.xlabel('Batch solver iterations')
plt.ylabel('RMS pose translational error (m)')

plt.figure()
plt.plot(e_pose_rms_r[1:])
plt.xlabel('Batch solver iterations')
plt.ylabel('RMS pose rotational error (rads)')
plt.show()

plt.figure()
plt.plot(fitness_vec)
plt.xlabel('Timestep')
plt.ylabel('Point cloud alignment fitness')
plt.show()

var_x = np.zeros((K+1))
var_y = np.zeros((K+1))
var_z = np.zeros((K+1))
var_th1 = np.zeros((K+1))
var_th2 = np.zeros((K+1))
var_th3 = np.zeros((K+1))

cov_final = np.linalg.inv(A)
var_final = np.diag(cov_final)
for i in range(len(dataset.frame_range)):
	var_x[i] = var_final[6*i]
	var_y[i] = var_final[6*i + 1]
	var_z[i] = var_final[6*i + 2]
	var_th1[i] = var_final[6*i + 3]
	var_th2[i] = var_final[6*i + 4]
	var_th3[i] = var_final[6*i + 5]

plt.figure()
plt.plot(range(K+1),e_pose_oneit[0,:],color='blue')
plt.plot(range(K+1),3*np.sqrt(var_x),color='red')
plt.plot(range(K+1),-3*np.sqrt(var_x),color='red')
plt.xlabel('Timestep')
plt.ylabel('Translational error along x-axis (m)')
plt.legend(['error','3 sigma confidence level'])

plt.figure()
plt.plot(range(K+1),e_pose_oneit[1,:],color='blue')
plt.plot(range(K+1),3*np.sqrt(var_y),color='red')
plt.plot(range(K+1),-3*np.sqrt(var_y),color='red')
plt.xlabel('Timestep')
plt.ylabel('Translational error along y-axis (m)')
plt.legend(['error','3 sigma confidence level'])

plt.figure()
plt.plot(range(K+1),e_pose_oneit[2,:],color='blue')
plt.plot(range(K+1),3*np.sqrt(var_z),color='red')
plt.plot(range(K+1),-3*np.sqrt(var_z),color='red')
plt.xlabel('Timestep')
plt.ylabel('Translational error along z-axis (m)')
plt.legend(['error','3 sigma confidence level'])

plt.figure()
plt.plot(range(K+1),e_pose_oneit[3,:],color='blue')
plt.plot(range(K+1),3*np.sqrt(var_th1),color='red')
plt.plot(range(K+1),-3*np.sqrt(var_th1),color='red')
plt.xlabel('Timestep')
plt.ylabel('Rotaional error about x-axis (rad)')
plt.legend(['error','3 sigma confidence level'])

plt.figure()
plt.plot(range(K+1),e_pose_oneit[4,:],color='blue')
plt.plot(range(K+1),3*np.sqrt(var_th2),color='red')
plt.plot(range(K+1),-3*np.sqrt(var_th2),color='red')
plt.xlabel('Timestep')
plt.ylabel('Rotaional error about y-axis (rad)')
plt.legend(['error','3 sigma confidence level'])

plt.figure()
plt.plot(range(K+1),e_pose_oneit[5,:],color='blue')
plt.plot(range(K+1),3*np.sqrt(var_th3),color='red')
plt.plot(range(K+1),-3*np.sqrt(var_th3),color='red')
plt.xlabel('Timestep')
plt.ylabel('Rotaional error about z-axis (rad)')
plt.legend(['error','3 sigma confidence level'])

plt.show()


