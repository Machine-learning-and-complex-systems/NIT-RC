import numpy as np
import networkx as nx
import os.path
import matplotlib.pyplot as plt
import time
import gc
gc.collect()
import warnings
warnings.filterwarnings("ignore")
import os
os.system('cls' if os.name == 'nt' else 'clear')
plt.rc('font',size=15)

'''
This code is to train the matrix W_{out} for noise we seperate in code 'RC for deterministic',
training phase is in code 'RC for deterministic',
predicting phase is in code 'Total'.
'''

UseDoubleWell=1
RCType=0# Different RC architecture: 0 deterministic RC to learn noise; 1/2 with noise in/out side tanh
dt = 0.01; #steplength
startLength = 2200 # Initial point of transition occuring
TestLength=300# Number of points for test/validation phase: 0.01 steplength
TrainLength= 580
start = time.time() 
"""SDE: Lorenz noise from code 'RC for deterministic'
"""

def DoubleWell():#Load data
    q=np.load('../data/1d bistable system with colored noise/noise.npy')
    U=q.reshape(1,-1)
    length = np.shape(U)[1]
    plt.plot(U[0,startLength-TrainLength:],color='coral',linestyle='-')
    plt.show()
    u=U[0,startLength-TrainLength:startLength + TestLength]
    u=u.reshape(1,-1)
    return u    
    

class Reservoir:
    def __init__(self,Utotal,hyperparameters):      
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

        
    def train(self,parascanRange):#To train W_{out}    
        self.Win =np.random.uniform(-self.kin,self.kin, (self.N, self.M + 1))
        self.Wb= np.random.uniform(-self.kin, self.kin, (self.N, 1))
        self.Wxi=np.random.uniform(-self.kin,self.kin, (self.N, self.M)) 
        # TODO: the values of non-zero elements are randomly drawn from uniform dist [-1, 1]
        g = nx.erdos_renyi_graph(self.N, self.D / self.N, 42, True)
        self.A = nx.adjacency_matrix(g).todense()
        ## spectral radius: rho  -> "appropriately scaled by its largest eigenvalue ρ."
        self.A = self.A *1.25/self.rho
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
                Xi= np.random.normal(0,1, (self.M, 1))#Gaussian white noise
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

    def train2(self,WoutPara):
        # use $W_{out}$ to get the deviation to approximate the noise distribution.
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
                  if j<0:#show the fitting to the noise distribution
                        plt.subplot(121)
                        plt.hist(NoiseTemp, Bins)
                        plt.subplot(122)
                        plt.hist(random_from_cdf, Bins)
                        plt.show()
                  
                  SampledNoise.append(random_from_cdf)
              self.SampledNoise.append(SampledNoise) 

    def _run(self,WoutPara,i):

        self.S = np.zeros((self.P, self.test_len))
        for index in range(self.NumBifurPara):
            V = np.vstack((x[self.train_len] for x in self.dataset[index]))
            self.r=self.rFinalTrainTotal[index]  
            if RCType==0:                
                global NoiseToUse
                NoiseToUse=np.array(self.SampledNoise)

            for t in range(self.test_len):
                # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + Win * V(t) + bias)
                if RCType==0:
                     self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.A,self.r) 
                              + np.dot(self.Win, np.vstack((self.bias, V)))+self.parak*np.dot(self.Wb, self.paralayer+self.parab))
              
                if np.isnan(np.sum(np.sum(np.array(self.r)))):
                    print('R')
                    print(np.max(V))
                    print(np.max(np.max(WoutPara)),np.min(np.min(WoutPara)))
                
                s = np.dot(WoutPara, np.vstack((self.bias, V, self.r)))             
                self.S[:, t] = np.squeeze(np.asarray(s))       
                              
                if RCType==0:                    
                    self.S[:, t] = self.S[:, t]                    
                V = s              
            plt.xlabel('Time step')
            dataset=np.array(self.dataset).flatten()
            plt.plot(dataset[self.train_len:],color='coral',linestyle='-')
            plt.plot(self.S[0,:],color='seagreen',linewidth=1,linestyle='-') 
            NoiseTotal[i,:]=self.S[0,:]
            #np.save('../data/1d bistable system with colored noise/NoiseLearned.npy',self.S[0,:])
            plt.savefig('../results/lorenz_noise.pdf',bbox_inches = 'tight')
            plt.show()

def GenerateEnsemble(Realizations):
    for j in range(Realizations):
        uTemp=DoubleWell() 
        if j==0:
            u=uTemp
        else:
            u=np.vstack((u, uTemp))
    return u
         
if __name__ == '__main__':
    
    if UseDoubleWell==1:
        L=50
        NoiseTotal=np.zeros((L,TestLength))
        for i in range (L):
            Realizations=1#00
            parascanRange=[0.2]
            BinNumber=4
            Utotal=[]

            
            rcpara=np.zeros(5)
            rcpara[0]=0.996    #K_{in}
            rcpara[1]=0.996  #D
            rcpara[2]=1.55    #relevant to rho: rho=0.806
            rcpara[3]=0.065   #alpha
            rcpara[4]=1e-7   #beta
        
            paraTest=parascanRange[0]#0.2
            Utotal=[]
            u=GenerateEnsemble(Realizations)
            Utotal.append(u)
            r2 = Reservoir(Utotal,rcpara)
            WoutTempReal = r2.train([paraTest])
            if RCType==0:
                    r2.train2(WoutTempReal)
            r2._run(WoutTempReal,i)
        #np.save('../data/1d bistable system with colored noise/NoiseTotal.npy',NoiseTotal)  #Save the noise we learnt for code 'Total'


