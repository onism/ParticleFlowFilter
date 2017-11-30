import numpy as np
from numpy import dot,eye
from numpy.linalg import inv
from  scipy.linalg import block_diag,sqrtm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from filterpy.monte_carlo import resampling
Nt = 4 # number of targets
T = 40 # time steps
brtho = 2  # measurement noise
btheta =  2*np.pi/180
Fk1 = np.array( [[1, 1, 0, 0],[0 ,1, 0, 0],[0 ,0,1 ,1],[0, 0, 0 ,1]])
Qk1 = np.diag([1,0.1,1,0.1])**2
Rk1 = np.diag([brtho,btheta])**2
Fk = Fk1
Qk = Qk1
Rk = Rk1
for i in range(Nt-1):
    Fk = block_diag(Fk,Fk1)
    Qk = block_diag(Qk,Qk1)
    Rk = block_diag(Rk,Rk1)

x_dim = 4 * Nt
z_dim = 2 * Nt

x_true = np.zeros((x_dim,T))
x_est = np.zeros((x_dim,T))
# produce true target
for t in range(0,T-1):
    x_true[:,[t+1]] = np.dot(Fk,x_true[:,[t]])   + sqrtm(Qk).dot(np.random.randn(x_dim,1))

# produce measurement
meas = np.zeros((z_dim,T))
for t in range(0,T):
    for i in range(Nt):
        meas[i*2,t] = np.sqrt(x_true[i*4,t]**2 + x_true[i*4+2,t]**2) + np.random.randn() * brtho
        meas[i*2 +1,t] = np.arctan2(x_true[i*4,t],x_true[i*4+2,t]) + np.random.randn() * btheta



def dhdx(Nt,x,x_dim,z_dim):
    Hs = []
    for i in range(Nt):
        p = np.array([ x[i*4], x[i*4+2]])
        mag =  x[i*4]**2 + x[i*4+2]**2
        sqrt_mag = np.sqrt(mag)
        H_tmp = np.array([[p[0]/sqrt_mag, 0, p[1]/sqrt_mag, 0],[p[1]/mag, 0, -p[0]/mag ,0 ]])
        Hs.append(H_tmp)
    Hs = np.asarray(Hs)
    H = Hs[0,:,:]
    for i in range(1,Nt):
        H = block_diag(H,Hs[i,:,:])
    return H
    #     H.append(H_tmp)
    # H = np.asarray(H)
    # return H


def GenerateLambda():
    """Exponentially distributed pseudo time steps"""
    delta_lambda_ratio = 1.2
    nLambda = 29 #number of exp spaced step sizes
    lambda_intervals = []
    for i in range(nLambda):
        lambda_intervals.append(i)

    lambda_1 = (1-delta_lambda_ratio)/(1-delta_lambda_ratio**nLambda)
    for i in range(nLambda):
        lambda_intervals[i] = lambda_1*(delta_lambda_ratio**i)

    return lambda_intervals



def dot4(A,B,C,D):
    """ Returns the matrix multiplication of A*B*C*D"""
    return dot(A, dot(B, dot(C,D)))

def dot3(A,B,C):
    """ Returns the matrix multiplication of A*B*C"""
    return dot(A, dot(B, C))

def caculate_flow_params(est,P,H,R,z,pmeasure,lam,x_dim):

    HPHt = dot3(H,P,H.T)
    A_inv = inv((lam*HPHt) + R)

    A = (-0.5)*dot4(P,H.T,A_inv,H)

    I2 = eye(x_dim)+(2*lam*A)
    I1 = eye(x_dim)+(lam*A)
    Ri = inv(R)

    IlamAHPHt = dot4(I1,P,H.T,Ri)
    err = pmeasure - dot(H,est)
    zdif = z - err
    b5 = dot(IlamAHPHt,zdif)
    Ax = dot(A,est)

    in_sum = b5 + Ax
    b = dot(I2,in_sum)
    return A,b


Np = 500
xp = np.random.multivariate_normal(x_true[:,0], Qk, Np).T # x_dim * Np
x_index_range = range(0,x_dim,4)
y_index_range = range(2,x_dim,4)
# plt.plot(xp[x_index_range,:], xp[y_index_range,:], 'x')
# plt.show()
w = 1.0/Np * np.ones((Np))

lam_vec = GenerateLambda()
nLambda = 29
for t in range(T):
    xp = np.dot(Fk,xp) + sqrtm(Qk).dot(np.random.randn(x_dim,Np))

    # particle flow
    P = np.cov(xp)
    lam = 0
    for j in range(nLambda):
        lam += lam_vec[j] #pseudo time step
        M = np.mean(xp, axis = 1)
        H = dhdx(Nt,M,x_dim,z_dim)
        y = np.zeros((z_dim))
        for i in range(Nt):
            y[i*2] = np.sqrt(M[i*4]**2 + M[i*4+2]**2)
            y[i*2 +1] = np.arctan2(M[i*4],M[i*4+2])
        A,b = caculate_flow_params(M,P,H,Rk,meas[:,t],y,lam,x_dim)
        dxdl = dot(A, xp) + np.tile(b,(Np,1)).T
        xp += (lam_vec[j]*dxdl)
    x_est[:,t] = np.mean(xp, axis = 1)






plt.figure('target')
for i in range(Nt):
    plt.plot(x_true[i*4,:],x_true[i*4+2,:],'ro-')
    plt.plot(x_est[i*4,:],x_est[i*4+2,:],'bx-')
plt.show()


