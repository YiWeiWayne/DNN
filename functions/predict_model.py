import numpy as np
import os
from functions import Data_load
def predict(head,model,model_name,window,stride,feat_size,big_frame_window,big_frame_stride):
    ans = []
    batch_size = 1 
    list_path = 'T'+head[1:]+'_'+str(window)+'_'+str(stride)+'_'+model_name
    if not os.path.exists(list_path):
        os.makedirs(list_path)
    (dataX, dataY, label_file, file_path)= Data_load.load_data(window,stride,feat_size,head)
    for i in range(0,len(dataX)):
        model.reset_states()
        tmp_x = [];tmp_y = [];
        tmp_x.append(dataX[i]);tmp_y.append(dataY[i])
        if big_frame_window !=0:
            (dataX_bframe, dataY_bframe, dataY_bframe_label)= Data_load.big_frame_extract(tmp_x,tmp_y,big_frame_window,big_frame_stride)
            X = np.asarray(dataX_bframe)
            output1 = []
            output1 = model.predict(X,batch_size = batch_size)
            ans = []
    #        for j in range(0,big_frame_window):
    #            ans.append(0)
            for j in range(0,len(output1)):
                if j == 0:
                    for k in range(0,big_frame_window):
                        if output1[j][k][0]>output1[j][k][1]:
                            ans.append(0)
                        else:
                            ans.append(1)
                else:
                    if output1[j][big_frame_window-1][0]>output1[j][big_frame_window-1][1]:
                        ans.append(0)
                    else:
                        ans.append(1)
        else:
            X = np.asarray(tmp_x[0])
            ans=[]
            for j in range(0,len(X)):
                output1 = []
                output1 = model.predict(np.reshape(X[j],(1,1,X.shape[1])),batch_size = batch_size)
                if output1[0][0][0]>output1[0][0][1]:
                    ans.append(0)
                else:
                    ans.append(1)
        file_name = os.path.split(file_path[i])[1][:-3]+'list'
        file_name = list_path+"/"+file_name
        print("Generate:",file_name)
        with open(file_name, "w") as fw:
            for s in ans:
                fw.write(str(s) +"\n")
            fw.close()