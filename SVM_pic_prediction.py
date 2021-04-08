import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
dir = 'E:\\courses\\Machine Learning\\SVM\pic\\PetImages'
categories = ['Cat','Dog']
data = []
for category in categories:
    path = os.path.join(dir,category)
    label= categories.index(category)
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        pet_img = cv2.imread(imgpath,0)
        try:
            pet_img = cv2.resize(pet_img,(50,50))
            image = np.array(pet_img).flatten()
            data.append([image,label])
        except Exception as e:
            pass

print(len(data))

pick_in = open('data1.pickle','wb')
pickle.dump(data,pick_in)        
pick_in.close()
pick_in =open('data1.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()
random.shuffle(data)
#print(data)
features = []
labels = []
for feature ,label in data:
    features.append(feature)
    labels.append(label)

x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.25)
model = SVC(kernel='rbf',random_state=0)
model.fit(x_train,y_train)

prediction = model.predict(x_test)
accuracy = model.score(x_test,y_test)
print('Accuracy : ',accuracy)
print('Prediction :',categories[prediction[0]])
mypet=x_test[0].reshape(50,50)
plt.imshow(mypet,cmap='gray')
plt.show()
