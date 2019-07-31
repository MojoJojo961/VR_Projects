with open('cifar-10-batches-py/data_batch_1', 'rb') as f:
    import pickle
    import numpy as np
    import cv2
    data = pickle.load(f, encoding='bytes')
    print(data.keys())
    dic = {}
    '''for i, d in enumerate(data[b'data']):
        print('{}: {} \t{} \t{}'.format(i,d,len(d), type(d)))
        img = np.array(d)
        print(img.shape)
        img = np.transpose(np.reshape(img,(3, 32,32)), (1,2,0))
        dic[d] = i
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break'''
    i = 1
    '''for cat, label in zip(data[b'data'], data[b'labels']):
        print('{} : {} : {}'.format(cat, label, i))
        i += 1
        img = np.array(cat)
        img = np.transpose(np.reshape(img,(3,32,32)), (1,2,0))
        print(img.shape)
        print(type(img))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        print(img.shape)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break'''
    #print(data['data']
    img = data[b'data']
    img = np.array(img[422])
    img = np.transpose(np.reshape(img, (3,32,32)), (1,2,0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    key, desc = sift.detectAndCompute(img, None)
    print(desc)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
