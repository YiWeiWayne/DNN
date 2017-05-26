from __future__ import print_function
import numpy as np
np.random.seed(1337)
#from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Masking
from keras.layers.recurrent import LSTM
import tensorflow as tf
import keras
from keras.utils import np_utils
import keras.backend.tensorflow_backend as KTF
from functions import Data_load
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

def get_session(gpu_fraction=0.25):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

'exception_verbosity = high'
batch_size = 1
hidden_units =256
nb_classes = 2
epochs = 10
print('Loading data...')
window = 80
stride = 40
feat_size = 26 
big_frame_window = 5
big_frame_stride = 1 
model_name = 'bigframes_w_'+str(big_frame_window)+'_s_'+str(big_frame_stride)+'_hiddenUnit_'+str(hidden_units)+'_epochs_'+str(epochs)
(dataX, dataY, label_file, file_path)= Data_load.load_data(window,stride,feat_size,'training_data')
(dataX_bframe, dataY_bframe, dataY_bframe_label)= Data_load.big_frame_extract(dataX,dataY,big_frame_window,big_frame_stride)

NG_files_indexes = [index for index in range(0,len(label_file)) if label_file[index]==1]
OK_files_indexes = [index for index in range(0,len(label_file)) if label_file[index]==0]
NG_big_frames_indexes = [index for index in range(0,len(dataY_bframe_label)) if dataY_bframe_label[index]==1]
OK_big_frames_indexes = [index for index in range(0,len(dataY_bframe_label)) if dataY_bframe_label[index]==0]

# make OK&NG big frames equal in size
np.random.shuffle(NG_big_frames_indexes)
np.random.shuffle(OK_big_frames_indexes)
big_frames_selected = []
for i in range(0,len(NG_big_frames_indexes)):
    big_frames_selected.append(NG_big_frames_indexes[i])
    big_frames_selected.append(OK_big_frames_indexes[i])
big_frames_selected = np.asarray(big_frames_selected)
X = []
Y = []
Y_big_frames = []
for i in big_frames_selected:
    X.append(np.asarray(dataX_bframe[i]))
    Y.append(np.asarray(dataY_bframe[i]))
    Y_big_frames.append(np.asarray(dataY_bframe_label[i]))
X = np.asarray(X);Y=np.asarray(Y);Y_big_frames=np.asarray(Y_big_frames)
Y = np_utils.to_categorical(Y)
Y = np.reshape(Y, (X.shape[0], X.shape[1], nb_classes))


# LSTM with Variable Length Input Sequences to Two Binary Output
model = Sequential()
model.add(LSTM(output_dim=hidden_units,batch_input_shape=(1,X.shape[1],X.shape[2]), return_sequences=True))
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
    checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
    history = model.fit(X, Y, batch_size=batch_size, nb_epoch=epochs, shuffle=True, callbacks=[earlyStopping], validation_split=0.1,verbose=1)
model.history = history
model.save('model/big_frame/'+model_name+'.h5')

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()