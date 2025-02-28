import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator, FixedFormatter,FuncFormatter
from matplotlib.ticker import StrMethodFormatter, ScalarFormatter, AutoMinorLocator
plt.rc('font',size=15)

ruptime=[]
rdowntime=[]
puptime=[]
pdowntime=[]
rupnumber=0
rdownnumber=0
pupnumber=0
pdownnumber=0

point=20
upstate=28
downstate=12
Exp=0

def Fsign(data):
    return np.where(data>point,1,np.where(data<point,-1,0))


def distribution(data,uptime,downtime):
    upnumber=0
    downnumber=0
    xshape=np.shape(data)[0]
    yshape=np.shape(data)[1]  
    index_up=0
    index_down=0
    length=max(xshape,yshape)
    diff=[]
    position=[]      
    if yshape > xshape: 
        data = data.T
    sign=Fsign(data)
        
    for i in range (length-1):
        diff.append(sign[i+1,0]-sign[i,0])
        if diff[i]!=0:
            position.append(i)
                
    for i in range (np.shape(sign)[0]):
        if sign[i,0]!=0:
            state=sign[i,0]
            index_up=i
            index_down=i
            break
            
    for i in range (len(position)):
        if i!=len(position)-1:
            if state<0:
                if diff[position[i]]>0:
                    if np.max(data[position[i]:position[i+1],0])>=upstate :
                        peroid=data[position[i]:position[i+1],0]
                        index_up=list(peroid>=upstate).index(True) + position[i]
                        uptime.append(index_up-index_down)
                        state=1
                        upnumber = upnumber+1
                        
                            
            elif state>0:
                if diff[position[i]]<0:
                    if np.min(data[position[i]:position[i+1],0])<=downstate : 
                        peroid=data[position[i]:position[i+1],0]
                        index_down=list(peroid<=downstate).index(True) + position[i]
                        downtime.append(index_down-index_up)
                        state=-1
                        downnumber = downnumber+1
                        
        else:
            if state<0:
                if diff[position[i]]>0:
                    if np.max(data[position[i]:,0])>=upstate :
                        peroid=data[position[i]:,0]
                        index_up=list(peroid>=upstate).index(True) + position[i]
                        uptime.append(index_up-index_down)
                        state=1
                        upnumber = upnumber+1
                        
            elif state>0:
                if diff[position[i]]<0:
                    if np.min(data[position[i]:,0])<=downstate : 
                        peroid=data[position[i]:,0]
                        index_down=list(peroid<=downstate).index(True) + position[i]
                        downtime.append(index_down-index_up)
                        state=-1
                        downnumber = downnumber+1
    return upnumber,downnumber
                        
def sci_formatter(x,pos):
    coeff, exp = "{:e}".format(x).split('e')
    coeff = float(coeff)
    if int(exp) < int(Exp):
        coeff = coeff /10
    return r"${:.2f}$".format(coeff)


                        
def draw_distribution(a,b,anumber,bnumber,title,xlable,bins_interval=100,margin=1):
    data=[a,b]
    left = min(np.min(a),np.min(b))
    right = max(np.max(a),np.max(b))
    bins= np.arange(math.floor(left), math.ceil(right), bins_interval)
    plt.clf()
    plt.xlim(math.floor(left) - margin, math.ceil(right) + margin)
    
    plt.xlabel(xlable) 
    plt.ylabel('Frequency')   
    hist,Bins,_ = plt.hist(data, bins=bins, density=True, color=['coral','seagreen'])
    plt.legend(['True: '+str(round(np.mean(a),2)),'Predicted: '+str(round(np.mean(b),2))],
                title='Average transition time',loc='upper right')
    plt.text(0.5, 1.03, '$T_{train}=25000$', transform=plt.gca().transAxes, ha='center',size=18)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_locator(FixedLocator(ax.get_yticks()))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    
    global Exp
    _, Exp = "{:e}".format(np.max(hist)).split('e')   
    
    formatter = FuncFormatter(sci_formatter)
    ax.yaxis.set_major_formatter(formatter)
    plt.annotate(r"$\times 10^{" + str(int(Exp)) + "}$",
             xy=(0, 1), xycoords='axes fraction',
             fontsize=ax.yaxis.get_major_ticks()[0].label1.get_fontsize(),
             verticalalignment='bottom')
    
    plt.savefig('../results/'+title + '.pdf',bbox_inches = 'tight')
    plt.show()


Realdata=np.load('../data/real data of protein folding/datafordistribution.npy')
realdata=Realdata[25000:125000]
realdata=realdata.reshape(1,-1)
predata=np.load('../data/real data of protein folding/predicted.npy')
predata=predata.reshape(1,-1)

rupnumber,rdownnumber=distribution(realdata,ruptime,rdowntime)
pupnumber,pdownnumber=distribution(predata,puptime,pdowntime)
draw_distribution(ruptime,puptime,rupnumber,pupnumber,'Transition_up time', 'uptime',bins_interval=280,margin=1)
draw_distribution(rdowntime,pdowntime,rdownnumber,pdownnumber,'Transition_down time', 'downtime',bins_interval=180,margin=1)