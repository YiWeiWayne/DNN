import os
from python_speech_features import mfcc
import wavio

#from scikits.talkbox.features import mfcc
window = 80
stride = 40
feat_size = 13 
def mfcc_extract(window,stride,feat_size):
    path = 'E:\\Wayne\\sound\\training_data\\'
    mfcc_feat = []
    for tops,dirs,files in os.walk(path):
        for f in files:
            if os.path.splitext(os.path.join(tops, f))[1] == '.wav':
                    w = wavio.read(os.path.join(tops, f))
                    tmp = mfcc(signal=w.data,samplerate=w.rate,winlen=window*0.001,winstep=stride*0.001,numcep=feat_size)
                    mfcc_feat.append(tmp)
    return mfcc_feat
