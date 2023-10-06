import numpy as np
import matplotlib.pyplot as plt
#s = np.random.poisson(5, 10000)
dt = 0.05               # timestep in day
tau = 7                 # timescale
No = 7                  # equilibrium average number of new storms generated per day
Nt = 10000              # number of integration time steps 
N = np.zeros(Nt)
A = np.zeros(Nt)
t = np.zeros(Nt)
#N = np.random.poisson(2.8,Nt)
#print(N)
N[0] = 0.
A[0] = 0.
for i in range(Nt-1):
    f1 = -np.random.poisson(N[i]/tau)+np.random.poisson(A[i]/tau)
    N[i+1] = N[i] + f1*dt
    f2 = (No-N[i])/tau
    A[i+1] = A[i] + f2*dt
    t[i+1] = t[i] + dt

#count, bins, ignored = plt.hist(s, 14, density=True)
#plt.scatter(x, y, s, c="g", alpha=0.5, marker=r'$\clubsuit$',label="Luck")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), tight_layout=True)
plt.subplot(2, 1, 1)
plt.plot(t[0:Nt],N[0:Nt],label="TCG events")
plt.xlabel("Time (day)")
plt.ylabel("Daily TCG events")
plt.legend(loc='upper left')
plt.xlim([100, 400])
plt.title('N time series')

plt.subplot(2, 1, 2)
plt.plot(t[0:8000],A[0:8000],label="TCG events")
plt.xlabel("Time (day)")
plt.ylabel("Daily TCG events")
plt.legend(loc='upper left')
plt.title('A time series')
plt.xlim([100, 400])
fig.tight_layout(pad=3.0)
#plt.show()

import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
Nd = np.zeros(round(Nt*dt))
for i in range(len(Nd)):
    Nd[i] = N[round(i*1./dt)]
plot_acf(Nd,lags=20)
plt.ylim([-0.5, 1])
plt.show()
