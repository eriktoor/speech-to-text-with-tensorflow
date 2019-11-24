#speech to text in tensor flow

import tflearn
import speech_data

learning_rate = 0.0001 #balanced between time and accuracy 
training_iters = 300000

batch = word_batch = speech_data.mfcc_batch_generator(64)
X, Y = next(batch)
trainX, trainY = X, Y  
testX, testY = X, Y  

net = tflearn.input_data([Non, 20, 80]) 
net = tflearn.lstm(net, 128, dropout=0.8) #dropout turns off data during training so nodes are forced to find new connections
net = tflearn.fully_connected(net, 10, activation="softmax")
net = tflearn.regression(net, optimizer="adam", learning_rate=learning_rate, loss="categorical_crossentropy")

model = tflearn.DNN(net, tensorboard_verbose=0)
