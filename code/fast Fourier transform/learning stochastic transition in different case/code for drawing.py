import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
plt.rc('font',size=15)
TestLength=400
L=50
Actual=np.load('../data/fast Fourier transform/actualTrajectory.npy')
predict=np.load('../data/fast Fourier transform/avgTrajectory.npy')
total=np.load('../data/fast Fourier transform/totalTrajectory.npy')
actual=np.zeros((1,TestLength))
actual[0,:]=Actual[0,:TestLength]

xtimes=np.linspace(36.6,40.6,400)
formatter = FormatStrFormatter("%.1f")
for i in range(L):
    plt.plot(xtimes,total[i,:],alpha=0.17,linewidth=2,linestyle='-')
plt.plot(xtimes,actual[0,:TestLength],color='coral',linestyle='-',linewidth=7,label='True')
plt.plot(xtimes,predict[0,:],color='seagreen',linestyle='-',linewidth=7,label='Predicted')
plt.legend(loc='upper left')
plt.gca().yaxis.set_major_formatter(formatter)
plt.savefig('../results/Total.pdf',bbox_inches='tight',dpi=600)
plt.show()
plt.clf()
ab=np.broadcast_to(actual,(L,TestLength))
results=ab-total
result=np.mean(results,axis=0)
# np.save('Error.npy',results)
# np.save('AvgError.npy',result)
for i in range(L):    
    plt.plot(xtimes,results[i,:],alpha=0.17,linewidth=2,linestyle='-')
plt.plot(xtimes,result,color='seagreen',linestyle='-',linewidth=7,label='Predicted')
plt.legend(loc='upper left')
ticks=np.linspace(-2,2,5)
plt.yticks(ticks)
plt.gca().yaxis.set_major_formatter(formatter)
plt.savefig('../results/Error.pdf',bbox_inches='tight',dpi=600)
plt.show()