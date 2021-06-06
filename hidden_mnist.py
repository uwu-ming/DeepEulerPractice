import tensorflow as tf
from tensorflow import keras

epochs=200
batch_size=128
verbose=1
nb_classes=10 # number of outputs
n_hidden=128
validation_split=0.2 # how much train is reserved for validation
mnist=keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

reshaped=784
x_train=x_train.reshape(60000,reshaped)
x_test=x_test.reshape(10000,reshaped)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')

y_train=tf.keras.utils.to_categorical(y_train,nb_classes)
y_test=tf.keras.utils.to_categorical(y_test,nb_classes)
#from keras.layers import Activation
model=tf.keras.models.Sequential()
model.add(keras.layers.Dense(n_hidden,input_shape=(reshaped,), activation="relu",name="layer_1"))
model.add(keras.layers.Dense(n_hidden, activation="relu",name="layer_2"))
model.add(keras.layers.Dense(nb_classes, activation="softmax",name="layer_3"))
model.summary()
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=128,epochs=50,verbose=1,validation_split=0.2)
test_loss,test_acc=model.evaluate(x_test,y_test)
print('Test accuracy:',test_acc)