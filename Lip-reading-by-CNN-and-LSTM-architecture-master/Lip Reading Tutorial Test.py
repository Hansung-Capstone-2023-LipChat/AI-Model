import numpy as np
import tensorflow as tf
import keras
import numpy as np
from numpy import argmax
import os
import cv2
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.preprocessing import image as image_utils

from keras.utils.generic_utils import CustomObjectScope
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model('Lib_Reading_10Frame_Model.h5')

    data = []
    label = []
    n_labels = 8
    Totalnb = 0
    data_nb = [0, 0, 0, 0, 0, 0, 0, 0]

    # Load Dataset
    for i in range(8):
        nb = 0
        for root, dirs, files in os.walk('normalized SVM/dataset/' + str(i + 1)):  # set directory
            for name in dirs:
                nb = nb + 1
        print(i, "Label number of Dataset is:", nb)
        Totalnb = Totalnb + nb
        data_nb[i] = data_nb[i] + nb
        for j in range(nb - 1):
            temp = []
            for k in range(10):
                name = 'normalized SVM/dataset/' + str(i + 1) + '/' + str(j + 1) + '/' + str(k + 1) + '.jpg'
                img = cv2.imread(name)
                res = cv2.resize(img, dsize=(img_col, img_row), interpolation=cv2.INTER_CUBIC)
                temp.append(res)
            label.append(i)
            data.append(temp)
    print("Total Number of Data is", Totalnb)
    Train_label = np.eye(n_labels)[label]
    Train_data = np.array(data)

    x = np.arange(Train_label.shape[0])
    np.random.shuffle(x)
    Train_label = Train_label[x]

    npp = 0
    n0 = 0
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    n5 = 0
    n6 = 0
    n7 = 0

    for i in range(Train_label.shape[0]):
        # Test model by each image timesteps
        Train_dataOne = Train_data[i]
        Train_dataOne = np.expand_dims(Train_dataOne, axis=0)
        # Use model class probability
        Y_data = model.predict(Train_dataOne, batch_size=1)
        # Use model class
        y_classes = Y_data.argmax(axis=-1)
        # print result predicted class / model input label ( One_hot Code )
        print(y_classes, Train_label[i])
        if (Train_label[i][y_classes] == 1):
            npp = npp + 1
            if (y_classes == 0):
                n0 = n0 + 1
            if (y_classes == 1):
                n1 = n1 + 1
            if (y_classes == 2):
                n2 = n2 + 1
            if (y_classes == 3):
                n3 = n3 + 1
            if (y_classes == 4):
                n4 = n4 + 1
            if (y_classes == 5):
                n5 = n5 + 1
            if (y_classes == 6):
                n6 = n6 + 1
            if (y_classes == 7):
                n7 = n7 + 1

    # print Total Result
    print("Number of correct answers is", npp, "into", 768, "Test Acc is", npp / 768)
    print("Number of correct answers for each classes is ", n0, n1, n2, n3, n4, n5, n6, n7)
    print("Accuary of each classes is ", n0 / data_nb[0], n1 / data_nb[1], n2 / data_nb[2], n3 / data_nb[3],
          n4 / data_nb[4], n5 / data_nb[5], n6 / data_nb[6], n7 / data_nb[7])

