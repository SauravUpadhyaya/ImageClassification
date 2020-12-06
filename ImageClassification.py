#Description: This program classifies images using CNN Model

#import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers 
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout 
from tensorflow.keras.layers import MaxPooling2D
from keras.utils import to_categorical
plt.style.use('fivethirtyeight')

#Load the data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Look at the data types of the variables
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

#get the shape of the arrays
print('Shape:', x_train.shape)
print('Shape:', y_train.shape)
print('Shape:', x_test.shape)
print('Shape:', y_test.shape)

#take a look at the first image as an array 
print(x_train[0])

#show image as a picture
index = 10
plt.imshow(x_train[index])

#image label
print('the image label is:', y_train[index])
print('the image label is:', y_test[index])

#Convert that label number to name
classification = ['aeroplane','automobile','bird','cat','deer','dog','frog','horse','ship', 'truck']
print('the name is:', classification[y_train[index][0]])

#Convert labels into numbers to input into neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#Print new labels
print('the new label is:', y_train_one_hot[index])
print('the new label is:', y_train_one_hot)

print(y_train_one_hot[index])
#print('the new label name is:', classification[y_train_one_hot[index][0]])

#Norrmalize pixls to be values betn 0 and 1
x_train = x_train/255
x_test = x_test/255

print(x_train)

#Creating the model architecture
model = Sequential ()
#Add the first convlution layer
model.add(Conv2D(32, (5, 5), activation = 'relu', input_shape = (32,32,3)))

#Add a pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

#Add second convolution layer
model.add(Conv2D(32, (5, 5), activation = 'relu'))


#Add another pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

#Add a flatten layer
model.add(Flatten())

#Add 1000 neurons to the model
model.add(Dense(1000, activation='relu'))

#Add drop layer
model.add(Dropout(0.5))

#Add 500 neurons  to the model
model.add(Dense(500, activation='relu'))

#Add drop layer
model.add(Dropout(0.5))

#Add 250 neurons  to the model
model.add(Dense(250, activation='relu'))

#Add 10 neurons  to the model
model.add(Dense(10, activation='softmax'))

#Compile the model
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

#Train the model
hist = model.fit(x_train, y_train_one_hot, 
           batch_size=256, epochs=10, validation_split=0.2 )

#Get the model accuracy
model.evaluate(x_test, y_test_one_hot)[1]

#Visualize the models accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

#Load the data
from google.colab import files # Use to load data on Google Colab
uploaded = files.upload() # Use to load data on Google Colab
new_image = plt.imread('Saurav_Certificate_Photo@AIT.jpg') #Read in the image (3, 14, 20)
img = plt.imshow(new_image)

#Resize image
from skimage.transform import resize
resized_image = resize(new_image, (32,32,3))
img = plt.imshow(resized_image)

#Predict
predictions = model.predict(np.array( [resized_image] ))
predictions

#PredictionSorted
list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions
for i in range(10):
  for j in range(10):
    if x[0][list_index[i]] > x[0][list_index[j]]:
      temp = list_index[i]
      list_index[i] = list_index[j]
      list_index[j] = temp
#Show the sorted labels in order from highest probability to lowest
print(list_index)

#Print 5 with highest probabilities
i=0
for i in range(5):
  print(classification[list_index[i]], ':', round(predictions[0][list_index[i]] * 100, 2), '%')

#To save this model 
model.save('my_Image_Classification_Model.h5')
