import tensorflow as tf
from tensorflow import keras
import numpy as np
from itertools import combinations as cb

epochs=200
batch_size=128
verbose=1
nb_classes=1 # number of outputs
n_hidden=80
validation_split=0.2 # how much train is reserved for validation
x=np.random.uniform(0,3,10)
x=np.sort(x,axis=0)
#x=np.linspace(0,3,10)
z=list(cb(x,2))
t=np.asarray(z)
f = lambda x,y: np.exp(x)
y0=0
y=np.zeros(len(x))
y[0]=y0
for i in range(0,len(x)-1):
    h=abs(x[i+1]-x[i])
    y[i+1]=y[i]+h*f(x[i],y[i])
w=np.zeros(len(t))
m=np.zeros(len(t))
for i in range (0,len(w)-1):
    for j in range(0,len(x)-1):
        if t[i,0]==x[j]:
            w[i]=y[j]
for i in range(0,len(m)-1):
    for j in range(0,len(x)-1):
        if t[i,1]==x[j]:
            m[i]=y[j]
m=m.reshape(-1,1)
w=w.reshape(-1,1)
th=np.hstack((t,w,m))
def R(x,y,w,z):
    dx=abs(w-x)
    return (z-w-dx*f(x,w))/(dx)**2
LTE=np.zeros(len(t))

for i in range(0,len(t)-1):
    LTE[i]=R(th[i,0],th[i,1],th[i,2],th[i,3])
#print(LTE)
x_train=th[0:45,0:3]
y_train=np.asarray(LTE)
model=tf.keras.models.Sequential()
model.add(keras.layers.Dense(nb_classes,input_dim=3, activation="relu",name="layer_1"))
model.add(keras.layers.Dense(n_hidden, activation="relu",name="layer_2"))
model.add(keras.layers.Dense(nb_classes, activation="softmax",name="layer_3"))
model.summary()
model.compile(optimizer='Adam',loss='mean_absolute_error',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=50,verbose=1)
#test_loss,test_acc=model.evaluate(x_test,y_test)
#print('Test accuracy:',test_acc)