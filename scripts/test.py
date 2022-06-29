#%%
import numpy as np
from sklearn.preprocessing import StandardScaler

#%%
data_not_scaled = np.arange(9).reshape(3,3)
print(data_not_scaled)
scaler0 = StandardScaler()
scaler1 = StandardScaler()
scaler2 = StandardScaler()
c0 = scaler0.fit_transform(data_not_scaled[:,0].reshape(data_not_scaled.shape[0],1))
c1 = scaler1.fit_transform(data_not_scaled[:,1].reshape(data_not_scaled.shape[0],1))
c2 = scaler2.fit_transform(data_not_scaled[:,2].reshape(data_not_scaled.shape[0],1))
#%%
data = np.column_stack((c0,c1,c2))
print(data)

print(scaler1.inverse_transform(data[:,1].reshape(-1,1)))
# %%
y = np.arange(30)

y = y.reshape((2,15),order ='F')
print(y.shape)
# %%
a = np.arange(10).T
b=a
print(a)
scalera = StandardScaler()
a = scalera.fit_transform(a.reshape(10,1))
print(a)
print(scalera.get_params(deep=True))
mean = np.mean(b)
std = np.std(b)
b=(b-mean)/std
print(b)
c = b*std + mean
print(c)
# %%
a = np.arange(10).T
print(a[5,:-1])
# %%
