import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import time
plt.rc('font',size=15)

start = time.time()
x = np.load('../data/data for previous approaches/data.npy')
data = np.zeros((10000,1))
data[:,0] = x[0,:10000]
window_size = 100

# parameters of RNN
units = 40
batch_size = 32

X = []
y = []
for i in range(len(data) - window_size):
    X.append(data[i:i+window_size])
    y.append(data[i+window_size])

X = np.array(X)
y = np.array(y)

# Apply LSTM to fit the data with noise
model = Sequential()
model.add(LSTM(units, input_shape=(window_size, 1)))
model.add(Dense(units))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=batch_size)

end = time.time()
input_sequence = data[-window_size:].reshape(1, window_size, 1)

# Predict 
predictions = []
for _ in range(300):
    prediction = model.predict(input_sequence)
    predictions.append(prediction[0, 0])
    input_sequence = np.append(input_sequence[:, 1:, :], prediction.reshape(1, -1, 1), axis=1)

# draw the result of prediction
xtimes = np.linspace(100,103,300)
plt.plot(xtimes, x[0,10000:10300],color = 'coral')
plt.plot(xtimes,predictions,color='seagreen')
plt.xticks(range(100,104))
plt.xlim(100,103)
plt.ylim(-2,2)

plt.xlabel('Time Step')
plt.ylabel('$u_1$')


plt.legend(['True','RNN'],loc='upper right')
plt.text(0.5, 1.03, 'units: '+str(units) + '  batch_size: '+ str(batch_size)  , transform=plt.gca().transAxes, ha='center',size=18)

plt.savefig('../results/'+str((units,batch_size))+'predict.pdf',bbox_inches='tight')
#np.save(str((units,batch_size))+'.npy', predictions)
plt.show()
#print('runtime:'+str(end - start))