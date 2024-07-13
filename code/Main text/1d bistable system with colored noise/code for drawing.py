import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',size=15)
TestLength=300
L=50
Actual=np.load('../data/1d bistable system with colored noise/actualTrajectory.npy')
predict=np.load('../data/1d bistable system with colored noise/avgTrajectory.npy')
total=np.load('../data/1d bistable system with colored noise/totalTrajectory.npy')
actual=np.zeros((1,TestLength))
actual[0,:]=Actual[0,:TestLength]

xtimes=np.linspace(22,25,300)
for i in range(L):
    plt.plot(xtimes,total[i,:],alpha=0.17,linewidth=2,linestyle='-')
plt.plot(xtimes,actual[0,:TestLength],color='coral',linestyle='-',linewidth=7,label='True')
plt.plot(xtimes,predict[0,:],color='seagreen',linestyle='-',linewidth=7,label='Predicted')
plt.legend(loc='upper left')
plt.savefig('../results/Total.pdf',bbox_inches='tight',dpi=600)
plt.show()
plt.clf()

ab=np.broadcast_to(actual,(L,TestLength))
results=ab-total
result=np.mean(results,axis=0)
for i in range(L):    
    plt.plot(xtimes,results[i,:],alpha=0.17,linewidth=2,linestyle='-')
plt.plot(xtimes,result,color='seagreen',linestyle='-',linewidth=7,label='Predicted')
plt.legend(loc='upper left')
ticks=np.linspace(-1.5,1.5,7)
plt.yticks(ticks)
plt.savefig('../results/Error.pdf',bbox_inches='tight',dpi=600)
plt.show()