import numpy as np
import networkx as nx
import os.path
import matplotlib.pyplot as plt
from scipy import signal
import time
import math
import gc
gc.collect()
import warnings
warnings.filterwarnings('ignore')
import os
import pysindy as ps
os.system('cls' if os.name == 'nt' else 'clear')
plt.rc('font',size=18)

def Sav_Gol(u):
    # import Savizky-Golay filter
    window_size = 50  # filter window size
    poly_order = 2  # polynomial order
    U = signal.savgol_filter(u, window_size, poly_order)
    return U

u = np.load('../data/data for previous approaches/data.npy')
K = Sav_Gol(u)
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