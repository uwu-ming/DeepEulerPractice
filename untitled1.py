import tensorflow as tf
from tensorflow import keras
import numpy as np

epochs=200
batch_size=128
verbose=1
nb_classes=10 # number of outputs
n_hidden=128
validation_split=0.2 # how much train is reserved for validation
x=np.random.uniform(0,3,20)
f = lambda x,y: np.exp(x)
y=np.zeros(len(x))
y=f(x,y)
z=np.zeros(len(x))
z[0]=x[len(x)-1]
for i in range(1,len(x)):
    z[i]=x[i-1]
w=np.zeros(len(x))
w=f(z,w)
#x_train=np.vstack([[x],[z],[y],[w]])
#print(x_train)
#print(x_train[1,3])