# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:48:51 2017

@author: wayne_chen
"""
import numpy as np
import os
from python_speech_features import mfcc
import wavio
import math  
def load_data(window,stride,feat_size):
    dataX = []
    dataY = []
    dataY_file_label = []
    tmp = []
    num_samples = []
    path = 'E:\\Wayne\\sound\\training_data\\'
    for tops, dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(os.path.join(tops, f))[1] == '.wav':
                w = wavio.read(os.path.join(tops, f))
                tmp = mfcc(signal=w.data,samplerate=w.rate,winlen=window*0.001,winstep=stride*0.001,numcep=feat_size)
                dataX.append(tmp)
                num_samples = tmp.shape[0]
                tmp = np.zeros(num_samples)
                if int(f[0:2])<=15:
                    t_f = open(os.path.join(tops, f)[:-4]+'.txt','r')
                else:
                    t_f = open(os.path.join(tops, f)[:-4]+'_human.txt','r')
                content_str = t_f.readlines()
                content = []             
                for i in content_str:
                    str_tmp = ''
                    for j in i:
                        if (j != '\n' and j != ' '):
                            str_tmp = str_tmp+j
                        if str_tmp!= '':content.append(int(str_tmp))
                if int(f[0:2])<15:
                    dataY.append(tmp)
                    dataY_file_label.append(0)
                elif int(f[0:2])==15:
                    start_t = math.floor(((content[44]-window)/stride)+1)
                    end_t = num_samples
                    tmp[start_t:end_t] = tmp[start_t:end_t]+1
                    dataY.append(tmp)
                    dataY_file_label.append(1)
                else:
                    index = 46
                    while index+1<len(content):
                        start_t = math.floor(((content[index]-window)/stride)+1)
                        end_t = math.ceil(((content[index+1]-window)/stride)+1)
                        tmp[start_t:end_t] = tmp[start_t:end_t]+1
                        index = index+2
                    dataY.append(tmp)
                    dataY_file_label.append(1)
    del tmp,num_samples,start_t,end_t,index,content_str,content,w,f
    return dataX, dataY,dataY_file_label