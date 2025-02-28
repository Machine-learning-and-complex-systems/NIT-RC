def DoubleWell(para):  #Generate dataset
    '''
    In this function, we generate the training and predicting sets from SDE (using Euler method).
    '''
    dimensions = 1
    a = 0;    b = 5;    c=0;    epsilon=0.3
    xtime = np.linspace(0, T2,int(T2/dt)+1)
    u = np.zeros((dimensions, int(T2/dt)+1))
    q = np.zeros((dimensions, int(T2/dt)))
    u[0,0]=1.5
    for i in range(int(T2/dt)):
        u[0,i + 1] = u[0,i] + dt * (-b*(-u[0,i]+u[0,i]**3+c))        
        q[0,i] =  np.sqrt(dt*2*epsilon*b) * np.random.randn()
        if i >=10000:
            anoise.append(q[0,i])
        u[0,i+1]=u[0,i+1]+q[0,i]
    plt.ylim(-2,2)
    plt.xlabel('Time(s)')
    plt.ylabel('$u_1$')
    plt.axhline(1,linestyle='--',color='navy',alpha=0.5)
    plt.axhline(-1,linestyle='--',color='navy',alpha=0.5)
    plt.plot(xtime,u[0,:],color='coral',linewidth=1,linestyle='-')
    plt.xlim(0,30)
    plt.savefig('atra.pdf',bbox_inches = 'tight')
    plt.show()
    u_t=np.zeros((dimensions,TestLength))
    u_t[0,:]=u[0,10001:int(T2/dt)+1]
    distribution(u_t,T_atimes,T_ainterval)
    return u   

def GenerateEnsemble(Realizations,para):
    for j in range(Realizations):
        uTemp=DoubleWell(para) 
        if j==0:
            u=uTemp
        else:
            u=np.vstack((u, uTemp))
    return u