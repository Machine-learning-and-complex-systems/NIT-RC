import numpy as np
import networkx as nx
import os.path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import time
import math
import gc
gc.collect()
import warnings
import pysindy as ps
from filterpy.kalman import KalmanFilter

warnings.filterwarnings('ignore')
import os
os.system('cls' if os.name == 'nt' else 'clear')
plt.rc('font',size=18)

def Kalman(u):
    # Import Kalman filter
    U=[]
    A = np.array([[1]])
    H = np.array([[1]])
    Q = np.array([[0.0005]])  # measurement noise covariance matrix
    R = np.array([[1]])    # measurement noise covariance matrix
    
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.F = np.array([[1.]])  # state transition matrix
    kf.H = np.array([[1.]])  # observation matrix
    kf.Q = Q               # measurement noise covariance matri
    kf.R = R               # measurement noise covariance matrix
    kf.x = np.array([[0.]])  
    kf.P = np.array([[1.]])  
    filtered_values = []
    for measurement in u[0,:]:
        kf.predict()
        kf.update(measurement)
        U.append(kf.x[0, 0])
    return U

u = np.load('../data/data for previous approaches/data.npy')
K = Kalman(u)
K=np.array(K)
K = K.reshape((-1,1))
U = np.zeros((10000,1))
U[:,0] = K[:10000,0]
m,n = U.shape

dt = 0.01  #time step
threshold=0.005  #lambda
opt = ps.STLSQ(threshold=threshold)
lib = ps.PolynomialLibrary(degree=3)

feature_names = ['x']
model = ps.SINDy(feature_names = feature_names , optimizer = opt , feature_library=lib)
model.fit(U, t=dt)
model.print()


