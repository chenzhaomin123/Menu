import numpy as np
import os
import cv2
from config import *
# traindir = '/home/chen/Videos/share/data/train_test/train_chinese_food/'
b=[]
g=[]
r=[]
for folder in os.listdir(traindir):
    folder_path = traindir + folder + '/'
    print folder_path
    for image in os.listdir(folder_path):
        image_path = folder_path + image
        img = cv2.imread(image_path) / 255.0
        b.append(img[:,:,0].mean())
        g.append(img[:,:,1].mean())
        r.append(img[:,:,2].mean())

b = np.stack(b)
g = np.stack(g)
r = np.stack(r)
f = open('mean_std','w')
print 'mean',b.mean(),g.mean(),r.mean()
print 'std',b.std(),g.std(),r.std()
print >>f,'mean',b.mean(),g.mean(),r.mean()
print >>f,'std',b.std(),g.std(),r.std()
f.close()
