import cv2
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_creat_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)
x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(x_train_scaled, y_train)

y_pred = clf.predict(x_test_scaled)
# accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is:- ", accuracy_score(y_test, y_pred))

cap = cv2.VideoCapture(0)

while(True):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        upper_left = (int(width/2-56), int(height/2-56))
        bottom_right = (int(width/2+56), int(height/2+56))
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        roi = gary[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        im_pil = Image.fromarray(roi)

        image_bw = im_pil.convert('L')
        image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)

        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20

        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)

        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted.min_pixel, 0, 255)

        max_pixel = np.max(image_bw_resized_inverted)
        
        image_bw_resized_inverted_scaled = np.asaarray(image_bw_resized_inverted_scaled)/max_pixel

        train_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
        test_pred = clf.predict(test_sample)
        print("Predict class is: ", test_pred)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            reak
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()