from __future__ import print_function
import numpy as np
np.random.seed(1337)
#from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
import tensorflow as tf
import keras
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
import keras.backend.tensorflow_backend as KTF
from functions import Data_load
import matplotlib.pyplot as plt
import time
tStart = time.time()#計時開始

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
epochs = 10
print('Loading data...')
window = 80
stride = 40
t_step = 1
feat_size = 26
state = False 
#data_base: 'NG+OK', 'NG_only', 'OK_only'
OK_files = 10
data_base = 'NG+OK'+str(OK_files)
#algorithm: 'LSTM', 'GRU', 'SimpleRNN'
alg_index=0
#for alg_index in range(0,2):
if alg_index ==0:
    if alg_index==0:
        algorithm = 'LSTM'
    elif alg_index==1:
        algorithm = 'GRU'
    elif alg_index==2:
        algorithm = 'SimpleRNN'
    model_name = algorithm+'_varied_length_max_len_'+str(max_len)+'_'+data_base+'_stateful_'+str(state)+'_w_'+str(window)+'_s_'+str(stride)+'_hiddenUnit_'+str(hidden_units)+'_epochs_'+str(epochs)
    (dataX, dataY, label_file, file_path)= Data_load.load_data(window,stride,feat_size,'training_data')
    #(dataX_bframe, dataY_bframe, dataY_bframe_label)= Data_load.big_frame_extract(dataX,dataY,big_frame_window,big_frame_stride)
    #extract frame_frame
    
    NG_files_indexes = [index for index in range(0,len(label_file)) if label_file[index]==1]
    OK_files_indexes = [index for index in range(0,len(label_file)) if label_file[index]==0]
    #NG_big_frames_indexes = [index for index in range(0,len(dataY_bframe_label)) if dataY_bframe_label[index]==1]
    #OK_big_frames_indexes = [index for index in range(0,len(dataY_bframe_label)) if dataY_bframe_label[index]==0]
#    OK_frames = 0
#    NG_frames = 0
#    for i in range(0,len(NG_files_indexes)):
#        NG_frames+=sum(dataY[NG_files_indexes[i]])
#        OK_frames+=len(dataY[NG_files_indexes[i]])-sum(dataY[NG_files_indexes[i]])
    np.random.shuffle(NG_files_indexes)
    np.random.shuffle(OK_files_indexes)
    file_selected = []
    if data_base == 'NG+OK':
        for i in range(0,len(NG_files_indexes)):
            file_selected.append(NG_files_indexes[i])
            file_selected.append(OK_files_indexes[i])
    elif data_base == 'NG_only':
        for i in range(0,len(NG_files_indexes)):
            file_selected.append(NG_files_indexes[i])
    elif data_base == 'OK_only':
        for i in range(0,len(OK_files_indexes)):
            file_selected.append(OK_files_indexes[i])
    elif data_base == 'NG+OK'+str(OK_files):
        for i in range(0,len(NG_files_indexes)):
            file_selected.append(NG_files_indexes[i])
            if i<OK_files:
                file_selected.append(OK_files_indexes[i])
    file_selected = np.asarray(file_selected)
    X = []
    Y = []
    Y_files = []
    sample_weight = []
    for i in file_selected:
        X.append(dataX[i])
        Y.append(dataY[i])
        sample_weight.append(np.ones(len(dataY[i])))
        Y_files.append(label_file[i])
    X = np.asarray(X);Y=np.asarray(Y);Y_files=np.asarray(Y_files);sample_weight = np.asarray(sample_weight)
    
    X = pad_sequences(X, maxlen=max_len, padding='post', dtype='float32')
    Y = pad_sequences(Y, maxlen=max_len, padding='post', dtype='int')   
    sample_weight = pad_sequences(sample_weight, maxlen=max_len, padding='post', dtype='int')   
    
    Y = np_utils.to_categorical(Y)
    Y = np.reshape(Y, (X.shape[0], max_len, nb_classes))
    
    
    
    # LSTM with Variable Length Input Sequences to Two Binary Output
    model = Sequential()
    #model.add(Masking(mask_value=0., batch_input_shape=(1,X.shape[1], X.shape[2])))
    if algorithm == 'LSTM':
        model.add(LSTM(output_dim=hidden_units,batch_input_shape=(1,X.shape[1],X.shape[2]), return_sequences=True, stateful = state))
    elif algorithm == 'GRU':
        model.add(GRU(output_dim=hidden_units,batch_input_shape=(1,X.shape[1],X.shape[2]), return_sequences=True, stateful = state))
    elif  algorithm == 'SimpleRNN':
       model.add(SimpleRNN(output_dim=hidden_units,batch_input_shape=(1,X.shape[1],X.shape[2]), return_sequences=True, stateful = state))
    #model.add(Reshape((X.shape[1:]), input_shape=(X.shape[1:])))
    #model.add(LSTM(output_dim=hidden_units, return_sequences=True, stateful = True))
    #model.add(LSTM(output_dim=hidden_units, return_sequences=True, stateful = True))
    #model.add(LSTM(output_dim=hidden_units, return_sequences=True, stateful = True))
    model.add(TimeDistributed(Dense(nb_classes)))
    model.add(Activation('softmax'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],sample_weight_mode = 'temporal')
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
    
    print("Train...")
    with tf.device('/gpu:0'):
        history = model.fit(X, Y, batch_size=batch_size, nb_epoch=epochs, callbacks=[earlyStopping], validation_split=0.1,verbose=1,shuffle = False,sample_weight = sample_weight)
    #        model.reset_states()
    model.save('model/varied_length/'+model_name+'.h5')
    print(history.history.keys())
    
    state = True
    model_name = algorithm+'_varied_length_max_len_'+str(max_len)+'_'+data_base+'_stateful_'+str(state)+'_w_'+str(window)+'_s_'+str(stride)+'_hiddenUnit_'+str(hidden_units)+'_epochs_'+str(epochs)
    
    model2 = Sequential()
    if algorithm == 'LSTM':
        model2.add(LSTM(output_dim=hidden_units,batch_input_shape=(1,1,X.shape[2]), return_sequences=True, stateful = state))
    elif algorithm == 'GRU':
        model2.add(GRU(output_dim=hidden_units,batch_input_shape=(1,1,X.shape[2]), return_sequences=True, stateful = state))
    elif  algorithm == 'SimpleRNN':
       model2.add(SimpleRNN(output_dim=hidden_units,batch_input_shape=(1,1,X.shape[2]), return_sequences=True, stateful = state))
    model2.add(TimeDistributed(Dense(nb_classes)))
    model2.add(Activation('softmax'))
    model2.summary()
    
    for nb, layer in enumerate(model.layers):
        model2.layers[nb].set_weights(layer.get_weights())
    model2.save('model/varied_length/'+model_name+'_test.h5')
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
    tEnd = time.time()#計時結束
    print("It cost %f sec" % (tEnd - tStart))#會自動做近位