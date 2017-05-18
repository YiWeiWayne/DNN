from __future__ import print_function
import numpy as np
import scipy.io as sio
np.random.seed(1337)
#from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Reshape
from keras.layers.recurrent import LSTM
import tensorflow as tf
import keras
import os
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
import keras.backend.tensorflow_backend as KTF
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import mfcc_extraction
import Data_load

def get_session(gpu_fraction=0.25):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

'exception_verbosity = high'
batch_size = 1
hidden_units =512
nb_classes = 2
max_len = 1000
epochs = 2
print('Loading data...')
window = 80
stride = 40
feat_size = 26  
feat = mfcc_extraction.mfcc_extract(window,stride,feat_size)
def load_data():
    dataX = []
    dataY = []
    Y = []
    X = []
    tmp = []
    num_samples = []
    path = "mfcc_out_80_40_26_mat\\training_data"
    for tops, dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(os.path.join(tops, f))[1] == '.mat':
    #            print(os.path.join(tops, f))
                test = sio.loadmat(os.path.join(tops, f))
                dataX.append(test['final_cep_data'])
                num_samples = test['final_cep_data'].shape[0]
#                file = open(os.path.join(tops, f)[:-3]+'txt')
#                result = []
#                tmp = []
#                for line in file:
#                    result.append(list(map(str,line.split('\n'))))
#                if result[44][0] == '1' or result[44][0] == '2':
                if os.path.split(os.path.join(tops, f))[1][:2] == '01' or os.path.split(os.path.join(tops, f))[1][:2] == '02':
                    tmp = np.zeros(num_samples)
                    dataY.append(tmp)
                else:
                    tmp = np.zeros(num_samples)+1
                    dataY.append(tmp)
    # convert list of lists to array and pad sequences if needed
    del tmp,num_samples
    X = pad_sequences(dataX, maxlen=max_len, padding='post', dtype='float32')
    Y = pad_sequences(dataY, maxlen=max_len, padding='post', dtype='int')
    # reshape X to be [samples, time steps, features]
#    X = np.reshape(X, (X.shape[0], max_len, X.shape[1]))   
    return X, Y
# reshape X to be [samples, time steps, features]
dataX,dataY,label_file= Data_load.load_data()
X = pad_sequences(dataX, maxlen=max_len, padding='post', dtype='float32')
Y = pad_sequences(dataY, maxlen=max_len, padding='post', dtype='int')   

Y = np_utils.to_categorical(Y)
Y = np.reshape(Y, (X.shape[0], max_len, nb_classes))
shape=(1,X.shape[1],X.shape[2])
# LSTM with Variable Length Input Sequences to Two Binary Output
model = Sequential()
model.add(LSTM(output_dim=hidden_units,batch_input_shape=(1,X.shape[1],X.shape[2]), return_sequences=True, stateful = True))
#model.add(Reshape((X.shape[1:]), input_shape=(X.shape[1:])))
#model.add(LSTM(output_dim=hidden_units, return_sequences=True, stateful = True))
#model.add(LSTM(output_dim=hidden_units, return_sequences=True, stateful = True))
#model.add(LSTM(output_dim=hidden_units, return_sequences=True, stateful = True))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')

print("Train...")
with tf.device('/gpu:0'):
    for i in range(epochs):
        history = model.fit(X, Y, batch_size=batch_size, nb_epoch=1, callbacks=[earlyStopping], validation_split=0.1,verbose=1)
        model.reset_states()
model.save('model/my_var_model_LSTM1_0_1split_adagrad_hid512_batch1_epoch_2_A2.h5')

print(history.history.keys())
##  "Accuracy"
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
## "Loss"
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()