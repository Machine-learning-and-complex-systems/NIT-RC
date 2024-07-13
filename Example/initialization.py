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

UseDoubleWell=1
RCType=0# Different RC architecture: 0 deterministic RC to learn noise; 1/2 with noise in/out side tanh
        # In our work, RCType=0
dt = 0.01; #steplength
TestLength=10000#00 # Number of points for test/validation phase: 0.01 steplength
T2 =100+TestLength*dt  # Total time length: including training phase and test/validation phase
T_atimes=[]
T_ptimes=[]
T_ainterval=[]
T_pinterval=[]
anoise=[]

start = time.time() 
"""SDE: double well
"""
#self.M is the ODE's dimension
#self.N is the number of reservoir node
#self.S is training data
#self.R is collection of reservoir state vectors: bias, reservoir variable, system variable
#self.r seems the reservoir variables