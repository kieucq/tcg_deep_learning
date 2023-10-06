import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
#
# readin data (3 diffenrent ways)
#with open('TC_genesis_CTL_5y.txt') as f:
#    contents = f.read()
#    print(contents)
tcg = []
for i in range(5):
    y1=np.loadtxt("TC_genesis_CTL_5y.txt")[:,i]
    tcg.append(y1)
tcg=np.reshape(tcg,len(y1)*5)
print(tcg.shape)
#
# doing fft now
#
a = tcg-np.mean(tcg)
Af = fft(a)
B  = np.abs(Af)**2
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), tight_layout=True)
plt.subplot(2, 1, 1)
plt.plot(B,label="Power spectrum of TCG events")
plt.xlabel("Wavenumber")
plt.ylabel("PSD")
plt.legend(loc='upper left')
#plt.xlim([100, 400])
plt.title('FFT')
plt.show()

