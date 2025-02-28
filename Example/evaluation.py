def distribution(data,T_times,T_interval):  #To evaluate the transition time and number
    '''
    Change the time series to three numbers: 
        -1 if <0
        0 if =0
        1 if >0
    and put these results in sign,
    '''
    xshape=np.shape(data)[0]
    yshape=np.shape(data)[1]  
    count=0
    index_up=0
    index_down=0
    length=max(xshape,yshape)
    diff=[]
    position=[]      
    if yshape > xshape: 
        data = data.T
    sign=np.sign(data)
    
    '''
    Consider where the symbol changed, and put these results in position,
    when changed, transition might happen.
    '''
    for i in range (length-1):
        diff.append(sign[i+1,0]-sign[i,0])
        if diff[i]!=0:
            position.append(i)
    
    '''
    Initialization
    '''            
    for i in range (np.shape(sign)[0]):
        if sign[i,0]!=0:
            state=sign[i,0]
            index_up=i
            index_down=i
            break
    
    '''
    Evaluation
    '''        
    for i in range (len(position)):
        if i!=len(position)-1:
            if state<0:
                if diff[position[i]]>0:
                    if np.max(data[position[i]:position[i+1],0])>=1 :
                        peroid=data[position[i]:position[i+1],0]
                        index_up=list(peroid>=1).index(True) + position[i]
                        T_interval.append(index_up-index_down)
                        state=1
                        count+=1
                                
            elif state>0:
                if diff[position[i]]<0:
                    if np.min(data[position[i]:position[i+1],0])<=-1 : 
                        peroid=data[position[i]:position[i+1],0]
                        index_down=list(peroid<=-1).index(True) + position[i]
                        T_interval.append(index_down-index_up)
                        state=-1
                        count+=1
        else:
            if state<0:
                if diff[position[i]]>0:
                    if np.max(data[position[i]:,0])>=1 :
                        peroid=data[position[i]:,0]
                        index_up=list(peroid>=1).index(True) + position[i]
                        T_interval.append(index_up-index_down)
                        state=1
                        count+=1
            elif state>0:
                if diff[position[i]]<0:
                    if np.min(data[position[i]:,0])<=-1 : 
                        peroid=data[position[i]:,0]
                        index_down=list(peroid<=-1).index(True) + position[i]
                        T_interval.append(index_down-index_up)
                        state=-1
                        count+=1
    T_times.append(count)    

    
def draw_distribution(a,b,title,xlable,bins_interval=1,margin=1):  #To draw the PDF of evaluation between test and predicted data
    '''
    This part is to draw the PDF between a and b
    '''
    data=[a,b]
    left = min(np.min(a),np.min(b))
    right = max(np.max(a),np.max(b))
    bins= np.arange(math.floor(left), math.ceil(right), bins_interval)
    plt.xlim(math.floor(left) - margin, math.ceil(right) + margin)
    plt.xlabel(xlable) 
    plt.ylabel('Frequency')   
    plt.hist(data, bins=bins, density=True, color=['coral','seagreen'])
    if xlable == 'Times' :
        plt.legend(['True: '+str(round(np.mean(a))),'Predicted: '+str(round(np.mean(b)))],title='Mean',fontsize=12,loc='upper right')
    else:
        plt.legend(['True: '+str(round(np.mean(a),2)),'Predicted: '+str(round(np.mean(b),2))],title='Mean' ,fontsize=12,loc='upper right')
    plt.savefig(title + '.pdf',bbox_inches = 'tight')
    plt.show()