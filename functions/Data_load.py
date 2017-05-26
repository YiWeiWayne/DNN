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
def load_data(window,stride,feat_size,head):
    dataX = []
    dataY = []
    dataY_file_label = []
    file_path = []
    tmp = []
    num_samples = []
    path = 'E:\\Wayne\\sound\\'+head+'\\'
    for tops, dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(os.path.join(tops, f))[1] == '.wav':
                file_path.append(os.path.join(tops, f))
                w = wavio.read(os.path.join(tops, f))
                tmp = mfcc(signal=w.data,samplerate=w.rate,winlen=window*0.001,winstep=stride*0.001,numcep=feat_size)
                dataX.append(tmp)
                num_samples = tmp.shape[0]
                if head is 'training_data':
                    if int(f[0:2])<=15:
                        t_f = open(os.path.join(tops, f)[:-4]+'.txt','r')
                    else:
                        t_f = open(os.path.join(tops, f)[:-4]+'_human.txt','r')
                else:
                    t_f = open(os.path.join(tops, f)[:-4]+'.txt','r')
                content_str = t_f.readlines()
                content = []             
                for i in content_str:
                    str_tmp = ''
                    for j in i:
                        if j.isnumeric():
                            str_tmp = str_tmp+j
                    if str_tmp!= '':
                        content.append(int(str_tmp))
                if int(f[0:2])<15:
                    tmp = np.zeros(num_samples)
                    dataY.append(tmp)
                    dataY_file_label.append(0)
                elif int(f[0:2])==15:
                    tmp = np.zeros(num_samples)
                    start_t = []
                    start_t = math.floor(((content[44]-window)/stride)+1)
                    end_t = num_samples
                    tmp[start_t:end_t] = tmp[start_t:end_t]+1
                    dataY.append(tmp)
                    dataY_file_label.append(1)
                elif int(f[0:2])>15:
                    tmp = np.zeros(num_samples)
                    start_t = []
                    index = 46
                    while index+1<len(content):
                        start_t = math.floor(((content[index]-window)/stride)+1)
                        end_t = math.ceil(((content[index+1]-window)/stride)+1)
                        tmp[start_t:end_t] = tmp[start_t:end_t]+1
                        index = index+2
                    dataY.append(tmp)
                    dataY_file_label.append(1)
    return dataX, dataY,dataY_file_label,file_path

def big_frame_extract(dataX,dataY,big_frame_window,big_frame_stride):
    dataX_bframe = []
    dataY_bframe = []
    dataY_bframe_label = []
    for i in range(0,len(dataX)):
        num_frames = len(dataX[i])
        num_big_frames = math.ceil(((num_frames-big_frame_window)/big_frame_stride)+1)
        for j in range(0,num_big_frames):
            if (j*big_frame_stride+big_frame_window)<=num_frames:
                dataX_bframe.append(dataX[i][j*big_frame_stride:j*big_frame_stride+big_frame_window])
                dataY_bframe.append(dataY[i][j*big_frame_stride:j*big_frame_stride+big_frame_window])
                if sum(dataY[i][j*big_frame_stride:j*big_frame_stride+big_frame_window])>0:
                    dataY_bframe_label.append(1)
                else:
                    dataY_bframe_label.append(0)
            else:
                dataX_bframe.append(dataX[i][num_frames-big_frame_window:num_frames])
                dataY_bframe.append(dataY[i][num_frames-big_frame_window:num_frames])
                if sum(dataY[i][num_frames-big_frame_window:num_frames])>0:
                    dataY_bframe_label.append(1)
                else:
                    dataY_bframe_label.append(0)
    return dataX_bframe,dataY_bframe,dataY_bframe_label