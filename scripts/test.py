#%%
import numpy as np
from sklearn.preprocessing import StandardScaler

#%%
array = np.zeros((10,2))
for i in range(10):
    for j in range(2):
        array[i-1][j-1]=i+j
print(array)
# %%
a = array[3:]
a = a [:4]
print(a)
# %%
