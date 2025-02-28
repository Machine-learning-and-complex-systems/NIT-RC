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
import pysindy as ps  #  a library of sindy-2016

warnings.filterwarnings('ignore')
import os
os.system('cls' if os.name == 'nt' else 'clear')
plt.rc('font',size=15)

# import data of id bistable system with white noise
u = np.load('../data/data for previous approaches/data.npy')
u = u.reshape(-1,1)
U = np.zeros((10000,1))
U[:,0] = u[:10000,0]
m,n = U.shape

# parameters of sindy
dt = 0.01  #time step
threshold=0.01  #lambda

# Create sindy 
opt = ps.STLSQ(threshold=threshold)
lib = ps.PolynomialLibrary(degree=3)

# fit data and print identification
feature_names = ['x']
model = ps.SINDy(feature_names = feature_names , optimizer = opt , feature_library=lib)
model.fit(U, t=dt)
model.print()

# draw the result
v=np.zeros((m,n))
w=np.zeros((m,n))
w[:,0] = u[10000:10000+m,0]

v[0,0]=w[0,0]
for i in range(m-1):
    v[i+1,0]=v[i,0] +dt * (-0.0191 -0.013 *v[i,0]**3)
    
# plt.plot(u[:,0])
xtimes = np.linspace(m*dt,2*m*dt,m)
plt.plot(xtimes,w[:,0],color = 'coral')
plt.plot(xtimes,v[:,0],color='seagreen')
plt.xticks(np.arange(100,201,10))
plt.xlim(100,130)
formatter = FormatStrFormatter("%.1f")
plt.gca().yaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
plt.ylim(-2,2)
plt.legend(['True','SINDy-2016'],loc='upper right')
plt.text(0.5, 1.03, 'Threshold = 0.01', transform=plt.gca().transAxes, ha='center',size=18)
plt.savefig('../results/Sindy.pdf',bbox_inches='tight')
plt.show()

