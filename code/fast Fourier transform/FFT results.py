import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
plt.rc('font',size=15)

dt = 0.01

# Protein data
realdata = np.load('../data/fast Fourier transform/datafordistribution.npy')
protein = realdata[0:7500]

protein_size = len(protein)
protein_fft = np.fft.rfft(protein) / protein_size

plt.clf()
plt.subplot(2, 1, 1)
plt.plot(protein)
plt.title('Input Signal')
plt.xlabel('Time')
plt.xticks(np.arange(0,7501,1500))
plt.xlim(0,7500)
plt.subplot(2, 1, 2)
plt.plot(np.abs(protein_fft),color = 'plum')
plt.title('FFT Result')
plt.xlabel('Frequency')
plt.xlim(0,30)
plt.tight_layout()
plt.savefig('../results/protein.pdf',bbox_inches = 'tight')
plt.show()
#----------------------------------------------------------------------

plt.clf()
# 1d bistable with white noise
U = np.load('../data/fast Fourier transform/data.npy')
bistable = U[0,:10000]

bistable_size = len(bistable)
bistable_fft = np.fft.rfft(bistable) / bistable_size

plt.subplot(2, 1, 1)
xtimes = np.linspace(0,int(bistable_size*dt),bistable_size)
plt.plot(xtimes,bistable)
plt.title('Input Signal')
plt.xlabel('Time')
plt.xticks(np.arange(0,int(bistable_size*dt)+1,20))
plt.xlim(0,100)
plt.subplot(2, 1, 2)
plt.plot(np.abs(bistable_fft),color = 'plum')
plt.title('FFT Result')
plt.xlabel('Frequency')
plt.xlim(0,200)
plt.tight_layout()
plt.savefig('../results/bistable.pdf',bbox_inches = 'tight')
plt.show()
#----------------------------------------------------------------------

plt.clf()
# mixed frequency 1
mix_size = 100000
M = np.load('../data/fast Fourier transform/lorenz.npy')
mix = M[:mix_size]
mix_fft = np.fft.rfft(mix) / mix_size

plt.subplot(2, 1, 1)
xtimes = np.linspace(0,int(mix_size*dt),mix_size)
plt.plot(xtimes,mix)
plt.title('Input Signal')
plt.xlabel('Time')
plt.xticks(np.arange(0,int(mix_size*dt)+1,200))
plt.xlim(0,200)
plt.subplot(2, 1, 2)
plt.plot(np.abs(mix_fft),color = 'plum')

plt.title('FFT Result')
plt.xlabel('Frequency')
plt.xlim(0,100)
plt.tight_layout()
plt.savefig('../results/mix.pdf',bbox_inches = 'tight')
plt.show()
#----------------------------------------------------------------------

plt.clf()
#mixed frequency 2
sep_size = 100000
S = np.load('../data/fast Fourier transform/test.npy')
sep = S[0,:sep_size]
sep_fft = np.fft.rfft(sep) / sep_size

plt.subplot(2, 1, 1)
xtimes = np.linspace(0,int(sep_size*dt),sep_size)
plt.plot(xtimes,sep)
plt.title('Input Signal')
plt.xlabel('Time')
plt.xticks(np.arange(0,int(sep_size*dt)+1,200))
plt.xlim(0,1000)
plt.subplot(2, 1, 2)
plt.plot(np.abs(sep_fft),color = 'plum')
plt.title('FFT Result')
plt.xlabel('Frequency')
plt.xlim(0,100)
plt.tight_layout()
plt.savefig('../results/seperated.pdf',bbox_inches = 'tight')
plt.show()

