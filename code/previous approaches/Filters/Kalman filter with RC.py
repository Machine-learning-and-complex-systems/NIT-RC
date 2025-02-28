import numpy as np
import networkx as nx
import os.path
import matplotlib.pyplot as plt
import time
import math
import gc
gc.collect()
import warnings
from filterpy.kalman import KalmanFilter
warnings.filterwarnings('ignore')
import os
os.system('cls' if os.name == 'nt' else 'clear')
plt.rc('font',size=18)

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
    

def DoubleWell(para):  #Generate dataset
    '''
    In this function, we generate the training and predicting sets from SDE (using Euler method).
    '''
    dimensions = 1
    xtime = np.linspace(0, T2,int(T2/dt)+1)

    u = np.load('../data/data for previous approaches/data.npy')
    
    U = Kalman(u)
    U=np.array(U)
    U=U.reshape((dimensions, int(T2/dt)+1))
    plt.plot(xtime,u[0,:],color='coral',alpha=0.8,linewidth=1.5,linestyle='-')
    plt.plot(xtime,U[0,:],color='blue',linewidth=1,linestyle='-')
    plt.xlim(0,10)
    plt.ylim(-2,2)
    plt.legend(['True','Filtered'],loc='upper right')
    plt.savefig('../results/Kalman.pdf',bbox_inches = 'tight')
    plt.show()
    plt.clf()
    plt.plot(xtime,U[0,:],color='blue',linewidth=1,linestyle='-')
    return U 

class Reservoir:
    def __init__(self,Utotal,hyperparameters):      
        '''
        Initialize the reservoir, input the training set, initialize some parameters, and input hyperparameters
        '''
        # Input layer
        u=Utotal[0]
        self.NumBifurPara=np.shape(Utotal)[0]       
        self.M = u.shape[0]  
        self.input_len = u.shape[1]
        self.dataset = Utotal

        # Reservoir layer
        self.N = 800
        self.dimensions = 1
        self.P = u.shape[0]
        self.bias = 0
        
        # Training relevant: have used their ratio of separating the input data
        self.init_len = 0
        self.train_len = np.int(self.input_len-TestLength-1)
        self.test_len = np.int(TestLength)
        self.error_len = np.int(TestLength)
        self.T2=self.input_len
        
        # Hyperparameters to be optimized: here set their initial value of hyperparameters
        self.kin =hyperparameters[0] # uniform distribution of $[-k_{in},k_{in}]$ of W_{in}, W_{b}, W_{\epsilon}
        self.D = hyperparameters[1] # degree of reservoir matrix A ...  #What is the degree function in their code: the degree of the renyi-Erdos network
        self.rho = hyperparameters[2] # spectral radius: rho  -> "appropriately scaled by its largest eigenvalue ρ."
        self.alpha =hyperparameters[3] 
        self.beta = hyperparameters[4] 
        self.parab=0 #set as zero for now, because no para-layer yet
        self.parak=0 #set as zero for now, because no para-layer yet
        self.paraepsilon=0 #epsilon for RC and consistent with SDE model

        
    def train(self,parascanRange): #To train the matrix W_{out}       
        '''
         Randomly generate Matrics W_{in}, W_{b}, and W_{xi} for input, 
         based on the uniform distribution in [-K_{in}, K_{in}],
         an ER network (g) with average degree D is to describe the connections of the nodes in reservoir,
         A is the adjacency matrix of g, and adjust its spectral radius by rho.
         This part is to train the output matrix W_{out} with the regression:
             W_{out}=\left(\mathbf{U R}^{\boldsymbol{\top}}\right) \cdot\left(\mathbf{R} \mathbf{R}^{\boldsymbol{\top}}+\beta\right)^{-1},
        '''
        self.Win =np.random.uniform(-self.kin,self.kin, (self.N, self.M + 1))
        self.Wb= np.random.uniform(-self.kin, self.kin, (self.N, 1))
        self.Wxi=np.random.uniform(-self.kin,self.kin, (self.N, self.M)) 
        # TODO: the values of non-zero elements are randomly drawn from uniform dist [-1, 1]
        g = nx.erdos_renyi_graph(self.N, self.D / self.N, 42, True)
        self.A = nx.adjacency_matrix(g).todense()
        # spectral radius: rho  -> "appropriately scaled by its largest eigenvalue ρ."
        self.A =self.A *1.25 /self.rho
        print(max(abs(np.linalg.eig(self.A)[0])))
        
        # run the reservoir with the data and collect r
        self.rFinalTrainTotal=[]
        for index in range(self.NumBifurPara):
            self.paralayer=parascanRange[index]
            self.R = np.zeros(
                    (1 + self.N + self.M, self.train_len-self.init_len ))
                # collection of input signals
            self.S = np.vstack((x[self.init_len + 1: self.train_len + 1] for x in self.dataset[index])) # make multi-dimensional input data from training window
            self.r = np.zeros((self.N, 1))
            for t in range(self.train_len):
                V = np.vstack((x[t] for x in self.dataset[index]))
                Xi= np.random.normal(0,1, (self.M, 1))
                if RCType==0:
                    #RC0: 0 deterministic RC to learn noise
                    self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.A,self.r) 
                          + np.dot(self.Win, np.vstack((self.bias, V)))+self.parak*np.dot(self.Wb, self.paralayer+self.parab))
                if RCType==1:
                    #RC1: noise inside tanh
                    self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.A,self.r) 
                          + np.dot(self.Win, np.vstack((self.bias, V)))+self.parak*np.dot(self.Wb, self.paralayer+self.parab)
                          + np.sqrt(self.paraepsilon)*np.dot(self.Wxi,Xi))            
                if RCType==2:
                    #RC2: noise outside tanh
                    self.r = (1 - self.alpha) * self.r + self.alpha * (np.tanh(np.dot(self.A,self.r) 
                         + np.dot(self.Win, np.vstack((self.bias, V)))+self.parak*np.dot(self.Wb, self.paralayer+self.parab))
                         + np.sqrt(self.paraepsilon)*np.dot(self.Wxi,Xi)) 
                if t >= self.init_len:
                    self.R[:, [t - self.init_len]
                           ] = np.vstack((self.bias, V, self.r))[:, 0]
            if index==0:
                self.RTotal=self.R
                self.STotal=self.S             
            else:            
                self.RTotal=np.append(self.RTotal,self.R, axis=1)
                self.STotal=np.append(self.STotal,self.S, axis=1)
            self.rFinalTrainTotal.append(self.r[:,-1])

        # train the output
        R = self.RTotal  # Transpose
        R_T=R.T        
        self.Wout = np.dot(np.dot(self.STotal, R_T), np.linalg.inv(
            np.dot(self.RTotal, R_T) + self.beta * np.eye(self.M + self.N + 1)))
        return self.Wout

    def train2(self,WoutPara): # To seperate noise
        '''
        This part is to seperate the noise (Training phase)
        '''
        global t
        self.SampledNoise = []
        for index in range(self.NumBifurPara):

            self.paralayer=parascanRange[index]
            self.R = np.zeros(
                    (1 + self.N + self.M, self.train_len - self.init_len))
            self.S = np.vstack((x[self.init_len + 1: self.train_len + 1] for x in self.dataset[index])) # make multi-dimensional input data from training window
            self.r = np.zeros((self.N, 1))
            for t in range(self.train_len):
                V = np.vstack((x[t] for x in self.dataset[index]))
             
                #RC0: 0 deterministic RC to learn noise
                self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.A,self.r) 
                          + np.dot(self.Win, np.vstack((self.bias, V)))+self.parak*np.dot(self.Wb, self.paralayer+self.parab))
                s = np.dot(WoutPara, np.vstack((self.bias, V, self.r)))
                self.S[:, t] = np.squeeze(np.asarray(s))                
            for i in range(self.dimensions):
              X=self.dataset[index][np.array(range(int(self.P/self.dimensions)))*self.dimensions+i, 1: self.train_len + 1]
              Y=self.S[np.array(range(int(self.P/self.dimensions)))*self.dimensions+i, 0: self.train_len ] ####    #self.dataset[range(self.P)*self.dimensions+i, self.train_len+1: self.train_len+self.error_len+1]
              SampledNoise=[]
              for j in range(X.shape[1]):
                  XX=np.array(X[:,j])  
                  YY=np.array(Y[:,j])
                  NoiseTemp=XX-YY
                  Bins=max(5,int(np.sqrt(X.shape[0])))
                  hist, bins = np.histogram(NoiseTemp, bins=Bins,range=(min(NoiseTemp),max(NoiseTemp)), density=True)#np.histogram(data, bins=50)
                  bin_midpoints = bins[:-1] + np.diff(bins)/2
                  cdf = np.cumsum(hist)
                  cdf = cdf / cdf[-1]
                  values = np.random.rand(X.shape[0])
                  value_bins = np.searchsorted(cdf, values)
                  random_from_cdf = bin_midpoints[value_bins]
                  if j<0: #show the fitting to the noise distribution
                        plt.subplot(121)
                        plt.hist(NoiseTemp, Bins)
                        plt.subplot(122)
                        plt.hist(random_from_cdf, Bins)
                        plt.show()
                  SampledNoise.append(random_from_cdf)
              self.SampledNoise.append(SampledNoise) 
        
    
    def _run(self,WoutPara,Load,Predict):
        '''
        run the trained ESN in alpha generative mode. no need to initialize here,
        because r is initialized with training data and we continue from there.

        The value of the switches are really important:
            Load=0 and Predict=0 to draw the deterministic part (slow-scale time series with different initial points),
            Load=1 and Predict=0 to draw the deterministic part and the distribution of the seperated noise,
                here, need to import the matrics saved when Load=0,
            Load=1 and Predict=0 to draw the predicted data and the distribution of the seperated noise,
            here, need to import the matrics saved when Load=0,
            and evaluation would run.
        '''
        
        if Load==0: # to save the matrics we generated
            np.save(r'Winw.npy',self.Win)
            np.save(r'Aw.npy',self.A)
            np.save(r'WoutParaw.npy',WoutPara)            
            np.save(r'Wb.npy',self.Wb)
            np.save(r'Wxi.npy',self.Wxi)
            np.save(r'r.npy',self.rFinalTrainTotal)
            
        else:  # to load the matrics we saved
            self.Win=np.load(r'Winw.npy')
            self.A=np.load(r'Aw.npy')
            WoutPara=np.load(r'WoutParaw.npy')
            self.Wb=np.load(r'Wb.npy')
            self.Wxi=np.load(r'Wxi.npy')
            self.rFinalTrainTotal=np.load(r'r.npy')
        
        # Predicted data
        self.S = np.zeros((self.P, self.test_len))
        
        # Input of the reservoir
        for index in range(self.NumBifurPara):
            V = np.vstack((x[self.train_len] for x in self.dataset[index]))
            
            # Reservoir states
            if Predict == 0:
                self.r=self.rFinalTrainTotal[index]
                
            else:
                self.r=self.rFinalTrainTotal[index]*0         
            
            if RCType==0:               
                global NoiseToUse
                NoiseToUse=np.array(self.SampledNoise)

            for t in range(self.test_len):
                # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + Win * V(t) + bias)
                if RCType==0:
                    #RC0: 0 deterministic RC to learn noise
                    self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.A,self.r) 
                          + np.dot(self.Win, np.vstack((self.bias, V)))+self.parak*np.dot(self.Wb, self.paralayer+self.parab))
                                          
                
                
                if np.isnan(np.sum(np.sum(np.array(self.r)))):
                    print('R')
                    print(np.max(V))
                    print(np.max(np.max(WoutPara)),np.min(np.min(WoutPara)))
                
                # Output of the reservoir
                s = np.dot(WoutPara, np.vstack((self.bias, V, self.r)))  
      
                
                self.S[:, t] = np.squeeze(np.asarray(s))
                              
                if RCType==0:
                   
                    self.S[:, t] = self.S[:, t]

                    
                xjp = np.random.randint(0, self.test_len-1) # Sample noise from distribution
                if Predict == 0:
                    V = s

                else:
                    V = s +  NoiseToUse[0,xjp,0]
                    

            
            K = np.zeros((self.dimensions, TestLength))
            Q = 0.0005
            K[0,0] = self.S[0,0]
            xtimes = np.linspace(int(T2-TestLength*dt),int(T2),TestLength)
            for i in range(TestLength - 1):
                if Q == 0.05:
                    K[0,i+1] = K[0,i] + dt * (-0.0161 - 0.014 * K[0,i] + 0.013 * K[0,i] **2 + 0.01* K[0,i]**3)
                elif Q == 0.005:
                    K[0,i+1] = K[0,i] + dt * (-0.0131 - 0.017 * K[0,i] + 0.012 * K[0,i] **2 + 0.015* K[0,i]**3)
                else:
                    K[0,i+1] = K[0,i] + dt * (-0.012 + 0.012* K[0,i]**3)
            plt.plot(xtimes, K[0,:], color = 'blueviolet', linewidth = 1, linestyle='-' )
            
            if Predict == 0:  
                plt.ylim(-2,2)
                x = np.linspace(self.test_len*dt,int(T2),self.test_len)
                plt.plot(x,self.S[0,:],color='seagreen',linewidth=1,linestyle='-') 
                plt.xticks(range(int(self.test_len*dt),int(T2)))
                plt.xlim(100,103)
                plt.ylim(-2,2)
                plt.legend(['Filtered','SINDy-2016','RC'],loc='upper right')
                plt.savefig('../results/predicted.pdf',bbox_inches = 'tight')
                plt.show()



def GenerateEnsemble(Realizations,para):
    for j in range(Realizations):
        uTemp=DoubleWell(para) 
        if j==0:
            u=uTemp
        else:
            u=np.vstack((u, uTemp))
    return u
                    
if __name__ == '__main__':
    
    if UseDoubleWell==1:
    
        for i in range (1):
            Realizations=1
            parascanRange=[0.2]
            para=np.zeros(4)
            para[0]=0  #a
            para[1]=5  #b 
            para[2]=0  #c 
            para[3]=0.3  #epsilon  
            BinNumber=4
            Utotal=[]
        
            rcpara=np.zeros(5)  # hyperparameters for RC
            rcpara[0]=4  #K_{in}
            rcpara[1]=4  #D
            rcpara[2]=4000  #relevant to rho: rho=0.0012
            rcpara[3]=0.15  #alpha
            rcpara[4]=1e-8  #beta
            
            paraTest=parascanRange[0]#0.2
            para[2]=paraTest
            Utotal=[]
            u=GenerateEnsemble(Realizations,para)
            Utotal.append(u)
            r2 = Reservoir(Utotal,rcpara)
            WoutTempReal = r2.train([paraTest]) #Acutally W_{out}
            
            Load=0  # Switch: 0 save matrics / 1 load saved matrics
            Predict=0  # Switch: 0 draw deterministic part / 1 draw predicted data
            if RCType==0:
                    r2.train2(WoutTempReal)
            r2._run(WoutTempReal,Load,Predict)
        

