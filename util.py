import numpy as np
from scipy.linalg import logm

def skew(p):
	p_skew = np.array([[0,-1*p[2],p[1]],[p[2],0,-1*p[0]],[-1*p[1],p[0],0]])

	return p_skew

def pose_inv_map(T_mat):

	w, v = np.linalg.eig(T_mat[0:3,0:3])
	

	index = np.argmin(np.absolute(w.imag))
	a = v[:,index].real

	phi = np.arccos( (0.5)*(np.trace(T_mat[0:3,0:3])-1) )
	phi = np.nan_to_num(phi)
	if phi != 0:
		lnT = logm(T_mat)
		th1 = lnT[2,1]
		th2 = lnT[0,2]
		th3 = lnT[1,0]
		rho = lnT[0:3,3]
	else:
		rho = T_mat[0:3,3]
		th1 = 0
		th2 = 0
		th3 = 0

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
	if phi != 0:
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
	else:
		T_mat = np.identity(4)
		T_mat[0:3,3:4] = t_vec[0:3]

	return T_mat

def invert_transform(T_inv):
	#build a transformation from its inverse (faster than T = np.linalg.inv(T_inv))

	T = np.identity(4)
	T[0:3,0:3] = T_inv[0:3,0:3].T
	T[0:3,3] = -1 * np.dot( T_inv[0:3,0:3].T , T_inv[0:3,3] )

	return T

def get_adjoint(T):
	#build an adjoint (Lie group - slide 47)

	T_Ad = np.identity(6)
	T_Ad[0:3,0:3] = T[0:3,0:3]
	T_Ad[3:6,3:6] = T[0:3,0:3]
	T_Ad[0:3,3:6] = np.dot(skew(T[0:3,3]), T[0:3,0:3])

	return T_Ad
