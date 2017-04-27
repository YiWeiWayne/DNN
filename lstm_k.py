from __future__ import print_function
import numpy as np
import scipy.io as sio
#from keras.optimizers import SGD
np.random.seed(1337)
#from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Reshape
from keras.layers.recurrent import LSTM
import tensorflow as tf
import keras
#from SpeechResearch import loadData

'exception_verbosity = high'
batch_size = 10
hidden_units =512
nb_classes = 2
print('Loading data...')

#get X_train,Y_train
test = sio.loadmat('input.mat')
X_train = test['input'];
X_train = X_train.astype('float32')

result=[]
file = open('1.txt')
for line in file:
     result.append(list(map(float,line.split(','))))
y1 = np.asarray(result)
Y_train = np.zeros((len(y1),2),int)
for i in range(0, len(y1)):
    if y1[i] == 1:
        Y_train[i][0] = 1
    if y1[i] == 2:
        Y_train[i][1] = 1    
#get X_test,Y_test
file_data = "E:/Wayne/lstm_code/keras/mfcc/T3/train_data.mat"
test = sio.loadmat(file_data)
X_test = test['train_data']
X_test = X_test.astype('float32')

file_data = "E:/Wayne/lstm_code/keras/mfcc/T3/train_label_frame.mat"
test = sio.loadmat(file_data)
y1 = test['train_label_frame']
y1 = y1.astype('int')
Y_test = np.zeros((len(y1),2),int)
for i in range(0, len(y1)):
    if y1[i] == 1:
        Y_test[i][0] = 1
    if y1[i] == 2:
        Y_test[i][1] = 1  

        
#X_test =X_train
#y_test = y_train
DATA_DIM = X_train.shape[1]
#(X_train, y_train), (X_test, y_test) = loadData.load_mfcc(10, 2)

print(len(X_train), 'train sequences')
#print(len(X_test), 'test sequences')
print('X_train shape:', X_train.shape)
#print('X_test shape:', X_test.shape)
print('Y_train shape:', Y_train.shape)
#print('y_test shape:', y_test.shape)
#print(y_test)
print('Build model...')

#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)
model = Sequential()
model.add(Reshape((DATA_DIM, 1), input_shape=(DATA_DIM,)))
model.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', activation='tanh', inner_activation='sigmoid'
               ,input_shape=(X_train.shape[1:]), return_sequences=True))
model.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', activation='tanh', inner_activation='sigmoid'
               , return_sequences=True))
model.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', activation='tanh', inner_activation='sigmoid'
               , return_sequences=True))
model.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', activation='tanh', inner_activation='sigmoid'
               ))
#model.add(Dense(156, activation="relu"))
#model.add(LSTM(39, activation="t"))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adagrad')
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

print("Train...")
with tf.device('/gpu:0'):
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=3, callbacks=[earlyStopping], validation_split=0.1, shuffle=False)
    score = model.evaluate(X_test, Y_test,batch_size=batch_size)
model.save('model/my_model_LSTM24_0_1split_adagrad_hid512_Thu1.h5')
print('Test score:', score)
#print('Test accuracy:', acc)

