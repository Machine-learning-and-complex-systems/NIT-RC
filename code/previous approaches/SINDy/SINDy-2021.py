import numpy as np
import networkx as nx
import os.path
import matplotlib.pyplot as plt
import time
import math
import gc
gc.collect()
import warnings
warnings.filterwarnings('ignore')
import os
os.system('cls' if os.name == 'nt' else 'clear')
plt.rc('font',size=15)

#Libraries for SINDY 2021, relevant files should put in the same path
from utils_NSS_SINDy import *
from scipy.linalg import qr
start = time.time()
U = np.load('../data/data for previous approaches/data.npy')
u = np.zeros((1,10000))
u[0,:]=U[0,:10000]
NoiseLevel=20

# Set a pin to generate new noise every run
pin=0

# parameters of sindy-2021
softstart=1
dt = 0.01
lam = 0.005 
libOrder=3   # order of polynomial function
N_iter=3    # iterations
disp=0       
NormalizeLib=0

# fit the noise data
Noise,smooth=approximate_noise(u,1)
Noise = np.transpose(Noise)
smooth=np.transpose(smooth)
print('E(xi): '+str(np.mean(Noise)))
print('D(xi): '+str(np.var(Noise)))

# smooth the data
dsmooth=CalDerivative(smooth,dt,1)
Theta=Lib(smooth,libOrder)

#fit the data
Xi=SINDy(Theta,dsmooth,lam,N_iter,disp,NormalizeLib)
end = time.time()
print(Xi)
