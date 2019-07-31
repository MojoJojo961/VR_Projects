import pickle
import numpy as np
import cv2
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='train a visual bag of words model')
    parser.add_argument('-d', help='path to the dataset', required=False)
    args = parser.parse_args()
    return args

args = parse_arguments()
label_dict = {}
with open('cifar-10-batches-py/batches.meta', 'rb') as f:
    data = pickle.load(f, encoding='UTF-8')
    for i in range(len(data['label_names'])):
        label_dict[i] = data['label_names'][i]

datasetpath = 'cifar-10-batches-py/'
p = datasetpath+args.d
print(p)
with open(p, 'rb') as f:

    data = pickle.load(f, encoding='bytes')
    i = 1
    for cat, label in zip(data[b'data'], data[b'labels']):
        img = np.array(cat)
        img = np.transpose(np.reshape(img, (3,32,32)), (1,2,0))

        if args.d=='test_batch':
            path = datasetpath+'test'
        else:
            path = datasetpath+label_dict[label]
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path+'/'+str(i)+'.jpg', img)
        i += 1
        if i==2000:
            break


