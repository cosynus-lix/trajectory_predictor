#%%
import numpy as np
#%%
x = np.zeros((100,2))
for k in range (101):
    x[k-1][0] = k
    x[k-1][1] = 100+k
s = np.sum(x[:,0][:3])
s
#%%