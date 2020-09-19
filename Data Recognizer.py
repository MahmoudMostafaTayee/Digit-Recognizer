# Digit recognizer competition
# kaggle.com competition
# https://www.kaggle.com/c/digit-recognizer/discussion/61480
# solved using CNN with 99% accuracy

# for any further questions, inquiries or communicating
# Mahmoud Mostafa Tayee
# mahmoud.tayee.1994@gmail.com

# libraries to be used
import os
import pandas as pd
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# importing the data file
data = pd.read_csv('D:\\Not yours\\Digit Recognizer\\train.csv')

# assigning the X and y from the imported file
Y = data.iloc[:, 0].values
x = data.iloc[:, 1:].values

# converting y from one column for all categories to one column for each category
y = np_utils.to_categorical(Y)

# scaling the x values as its values from 0 to 255
X = x/255

# splitting the data to train and test with ratio 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



#Reshape the data to handle it like pictures with 28*28 
X_train_reshaped = X_train.reshape(-1,28,28,1)
X_test_reshaped = X_test.reshape(-1,28,28,1)

# create a convolutional neural network model
CNN_model = Sequential()

# 1st convolutional layer
CNN_model.add(Conv2D(32, (3,3),input_shape = (28,28,1), activation = 'relu' ))
CNN_model.add(MaxPool2D(pool_size=(2,2)))

# 2nd convolutional layer
CNN_model.add(Conv2D(32, (3,3), activation = 'relu' ))
CNN_model.add(MaxPool2D(pool_size=(2,2)))

CNN_model.add(Dropout(0.2))
CNN_model.add(Flatten())

# fully connected network
CNN_model.add(Dense(128, activation = 'relu'))
CNN_model.add(Dense(y.shape[1], activation = 'softmax'))

#compile model
CNN_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# fitting the model
CNN_model.fit(X_train_reshaped, y_train, validation_data=(X_test_reshaped, y_test), epochs=10, batch_size=100, verbose=2)

# evaluating the model
scores = CNN_model.evaluate(X_test_reshaped, y_test, verbose = 0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# reading the data to predict then submit
X_to_predict = pd.read_csv('D:\\Not yours\\Digit Recognizer\\test.csv')
X_to_predict = np.array(X_to_predict)

# reshaping the data
X_to_predict = X_to_predict.reshape(-1,28,28,1)

# predicting the results
y_predicted = CNN_model.predict(X_to_predict)

# getting the number that had been predicted by the model for every record
# it's the step of inverting np_UTILS.to_categorical()
y_predicted = pd.DataFrame(y_predicted)
final_value = y_predicted.idxmax(axis = 1) 

# getting the data in the format to be assigned in.
final_value = pd.DataFrame(final_value)
y_pred_index = final_value.index + 1
y_pred_index = pd.DataFrame(y_pred_index)
assigned_format = pd.concat([y_pred_index , final_value], axis = 1)
assigned_format.columns = ['ImageId', 'Label']

# saving the result as csv file to be assigned
assigned_format.to_csv('D:\\Not yours\\Digit Recognizer\\submit.csv', index = False)

