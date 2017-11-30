import numpy as np
from numpy import eye, zeros,dot
from  scipy.linalg import block_diag,sqrtm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from filterpy.monte_carlo import resampling
Nt = 2 # number of targets
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


# plt.figure('target')
# for i in range(Nt):
#     plt.plot(x_true[i*4,:],x_true[i*4+2,:],'r-')
# plt.show()

# plt.figure('measurement')
# for i in range(Nt):
#     plt.plot(meas[i*2,:],meas[i*2+1,:],'r-')
# plt.show()


# particle filter
# initilization particles
Np = 5000
xp = np.random.multivariate_normal(x_true[:,0], Qk, Np).T # x_dim * Np
x_index_range = range(0,x_dim,4)
y_index_range = range(2,x_dim,4)
# plt.plot(xp[x_index_range,:], xp[y_index_range,:], 'x')
# plt.show()
w = 1.0/Np * np.ones((Np))


for t in range(T):
    xp = np.dot(Fk,xp) + sqrtm(Qk).dot(np.random.randn(x_dim,Np))
    # plt.plot(xp[x_index_range,:], xp[y_index_range,:], 'x')
    # plt.show()
    y = np.zeros((z_dim,Np))
    for i in range(Nt):
        y[i*2,:] = np.sqrt(xp[i*4,:]**2 + xp[i*4+2,:]**2)
        y[i*2 +1,:] = np.arctan2(xp[i*4,:],xp[i*4+2,:])

    #     plt.plot(meas[i*2,:],meas[i*2+1,:],'x')
    #     plt.plot(meas[i*2,t],meas[i*2+1,t],'ro')
    # plt.show()
    for i in range(Np):
        w[i] = w[i] * multivariate_normal.pdf(y[:,i], mean=meas[:,t], cov=Rk)

    print np.sum(w)
    w = w/np.sum(w)

    index = resampling.systematic_resample(w)
    xp = xp[:,index]
    w = 1.0/Np * np.ones((Np))
    x_est[:,t] = np.mean(xp, axis = 1)






plt.figure('target')
for i in range(Nt):
    plt.plot(x_true[i*4,:],x_true[i*4+2,:],'ro-')
    plt.plot(x_est[i*4,:],x_est[i*4+2,:],'bx-')
plt.show()


