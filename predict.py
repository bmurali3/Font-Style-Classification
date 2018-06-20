
"""
Utility for predicting on images
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random

from model import *

#tf.logging.set_verbosity(tf.logging.INFO)


def predict(eval_data, eval_labels, encoder):
    with tf.Session() as sess:
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
        mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="mnist_convnet_model3")
        eval_results = mnist_classifier.predict(input_fn=eval_input_fn, checkpoint_path="mnist_convnet_model3/model.ckpt-10001")
        eval_preds = []
        for x, each in enumerate(eval_results):
            eval_preds.append(int(each["classes"]))

        eval_preds = encoder.inverse_transform(eval_preds)

        # plot random image with prediction
        i = random.randint(0, len(eval_data)-1)
        plot_image(eval_data[i], "Prediction: "+eval_preds[i])

        binary_preds = [i.split('-')[0] for i in eval_preds]
        binary_gt = [i.split('-')[0] for i in encoder.inverse_transform(eval_labels).tolist()]

        print(classification_report(y_true=binary_gt, y_pred=binary_preds))
        print("Accuracy: ", accuracy_score(y_true=binary_gt, y_pred=binary_preds))

def read_folder(dir_name):

    values = []
    labels = []
    
    for img in os.listdir(dir_name):
        file = os.path.join(dir_name, img)
        if not file.endswith("png"):
            continue
        x = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if x is None:
            print("Unable to read image", img)
            continue
        x_small = cv2.resize(x, (32,32), interpolation = cv2.INTER_AREA)
        values.append(x_small.flatten())
        labels.append("-".join([img.split(".")[0].split("-")[i].lower() for i in [0,2]]))
    
    data = np.array(values, dtype=np.float16)

    target = np.array(labels)
    return data, target

def plot_image(img, title):
    plt.imshow(np.reshape(img, (32,32)).astype(int), cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title(title)
    plt.show()

if __name__ == '__main__':
# Load training and eval data
# Two classes have been split approximately 3:4 with few fonts held out for testing

    x_test1, y_test1 = read_folder('../data/serif/test')
    x_test2, y_test2 = read_folder('../data/sansserif/test')

    eval_data = np.append(x_test1, x_test2, axis = 0)
    eval_labels = np.append(y_test1, y_test2, axis = 0)

    # load encoder, encoder maps 'serif-[a-z]', 'sansserif[a-z]' to indices [0-51]
    encoder = LabelEncoder()
    encoder.classes_ = np.load('classes.npy')

    eval_labels = encoder.transform(eval_labels)
    predict(eval_data, eval_labels, encoder)

