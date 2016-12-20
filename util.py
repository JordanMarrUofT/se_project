import numpy as np

def pose_inv_map(T_mat):

	w, v = np.linalg.eig(T_mat[0:3,0:3])
	

	index = np.argmin(np.absolute(w.imag))
	a = v[:,index].real

	phi = np.arccos( (0.5)*(np.trace(T_mat[0:3,0:3])-1) )

	a_skew = np.array([[0,-1*a[2],a[1]],[a[2],0,-1*a[0]],[-1*a[1],a[0],0]])

	J = (1/phi)*np.sin(phi)*np.identity(3)
	J = J + (1 - (1/phi)*np.sin(phi))*np.dot( a , a.T )
	J = J + (1/phi)*(1 - np.cos(phi))*a_skew

	rho = np.dot( np.linalg.inv(J) , T_mat[0:3,3] )
	th1 = phi*a[0]
	th2 = phi*a[1]
	th3 = phi*a[2]

	t_vec = np.array([[rho[0]],[rho[1]],[rho[2]],[th1],[th2],[th3]])

	return t_vec

def rot_for_map(r_vec):

	#rotation forward mapping
	a = r_vec#column of rotational deltas
	phi = np.linalg.norm(a)
	a = (1/phi)*a
	a_skew = np.array([[0,-1*a[2],a[1]],[a[2],0,-1*a[0]],[-1*a[1],a[0],0]])

	R_mat = np.cos(phi)*np.identity(3)
	R_mat = R_mat + (1-np.cos(phi))*np.dot( a , a.T )
	R_mat = R_mat + np.sin(phi)*a_skew

	return R_mat

def pose_for_map(t_vec):

	#pose forward mapping
	a = t_vec[3:6]#column of rotational deltas
	phi = np.linalg.norm(a)
	a = (1/phi)*a
	a_skew = np.array([[0,-1*a[2],a[1]],[a[2],0,-1*a[0]],[-1*a[1],a[0],0]])

	C = np.cos(phi)*np.identity(3)
	C = C + (1-np.cos(phi))*np.dot( a , a.T )
	C = C + np.sin(phi)*a_skew

	J = (1/phi)*np.sin(phi)*np.identity(3)
	J = J + (1 - (1/phi)*np.sin(phi))*np.dot( a , a.T )
	J = J + (1/phi)*(1 - np.cos(phi))*a_skew

	r = np.dot( J , t_vec[0:3] )

	T_mat = np.identity(4)
	T_mat[0:3,0:3] = C
	T_mat[0:3,3:4] = r

	return T_mat

def invert_transform(T_inv):
	#build a transformation from its inverse (faster than T = np.linalg.inv(T_inv))

	T = np.identity(4)
	T[0:3,0:3] = T_inv[0:3,0:3].T
	T[0:3,3] = -1 * np.dot( T_inv[0:3,0:3].T , T_inv[0:3,3] )

	return T
