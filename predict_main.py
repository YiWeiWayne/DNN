# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:23:39 2017

@author: wayne_chen
"""

from keras.models import load_model
from functions import predict_model
window = 80
stride = 40
feat_size = 26 

big_frame_window = 0
big_frame_stride = 0
state = True
max_len = 1000
hidden_units =512
epochs = 10
#data_base: 'NG+OK', 'NG_only', 'OK_only'
OK_files = 10
data_base = 'NG+OK'+str(OK_files)
#algorithm: 'LSTM', 'GRU', 'simpleRNN'

algorithm = 'LSTM'
model_name = algorithm+'_varied_length_max_len_'+str(max_len)+'_'+data_base+'_stateful_'+str(state)+'_w_'+str(window)+'_s_'+str(stride)+'_hiddenUnit_'+str(hidden_units)+'_epochs_'+str(epochs)
model = load_model('model/varied_length/'+model_name+'_test.h5')

#big_frame_window = 5
#big_frame_stride = 1
#model_name = 'bigframes_w_'+str(big_frame_window)+'_s_'+str(big_frame_stride)+'_hiddenUnit_256_epochs_10'
#model = load_model('model/big_frame/'+model_name+'.h5')
                  
#head = 'testing2'
#predict_model.predict(head,model,model_name,window,stride,feat_size,big_frame_window,big_frame_stride)
#head = 'testing3'
#predict_model.predict(head,model,model_name,window,stride,feat_size,big_frame_window,big_frame_stride)
head = 'testing3_plus'
predict_model.predict(head,model,model_name,window,stride,feat_size,big_frame_window,big_frame_stride)
head = 'testing5'
predict_model.predict(head,model,model_name,window,stride,feat_size,big_frame_window,big_frame_stride)