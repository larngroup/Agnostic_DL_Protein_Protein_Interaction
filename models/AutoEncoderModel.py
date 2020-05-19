
import tensorflow as tf
import random
import keras
import numpy as np
import csv
import os
from tensorflow import keras
from keras import *
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.callbacks import *
from keras.optimizers import *
from sklearn.model_selection import StratifiedKFold
from keras.initializers import glorot_uniform
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from sklearn.metrics import roc_auc_score, confusion_matrix



	
class AutoEncoderModel:

	def __init__(self, paddVal, activationFun, neurons, batch, epochs, learnRate, path1, path2, aaSize, opt, dataToLoad):
		# Defyning the instance variables
		# my-1367 or teacher-1267
		self.paddVal=paddVal
		if activationFun=='relu':
			self.activationFun='relu'
		else:
			get_custom_objects().update({'swish':Swish(swish)})
			self.activationFun='swish'
		self.neurons=neurons
		self.learnRate=learnRate
		if opt==1:
		# The optimizer is Adam because, accordingly to literature is the one that gets the best results
			self.optimizer=Adam(lr=learnRate)
		elif opt==2:
			self.optimizer=SGD(lr=learnRate, decay=0, momentum=0.5)
		else:
			self.optimizer=RMSprop(lr=learnRate)
		self.batch=batch
		self.epochs=epochs
		# Path to save the model architecture and the weights in a .h5 file
		self.path1=path1
		self.path2=path2
		# Number of aminoacides wanted, either 7 or 21
		self.aaSize=aaSize
		'''
		According to the architecture that u want to hyperparameter tuning, unccomment the one you prefer
		1) Global:
				[Convolutional]^n -> GlobalMaxPooling ->[FC]^m -> OutputLayer
		2) Gap:
				[Convolutional -> MaxPooling]^n -> GlobalAveragePooling -> OutputLayer
		3) Local (the standard approach):
				[Convolutional -> MaxPooling]^n -> [FC]^m -> OutputLayer
		4) ResNet
		'''
		self.dataToLoad=dataToLoad
		# Callbacks
		self.earlyStop=EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=0, mode='max', baseline=None,restore_best_weights=True)
		self.checkpoint1=ModelCheckpoint(filepath=self.path1,monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
		self.checkpoint2=ModelCheckpoint(filepath=self.path2,monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
		if self.aaSize==7:
			self.encode={"A": 1, "R": 5, "N": 4, "D": 6, "C": 7, "E": 6, "Q": 4, "G": 1, "H": 4, "I": 2, "L": 2, "K": 5, "M": 3, "F": 2, "P": 2, "S": 3, "T": 3, "W": 4, "Y": 3, "V": 1, "U": 7}
		else:
			self.encode={"C": 1, "E": 2, "L": 3, "T": 4, "I": 5, "P": 6, "A": 7, "W": 8, "R": 9, "G": 10, "V": 11, "M": 12, "H": 13, "F": 14, "Y": 15, "Q": 16, "N": 17, "D": 18, "S": 19, "K": 20, "U": 21}
		# Loading the train and test data, shuffling it and encoding it
		if dataToLoad==1:
			self.trainA,self.trainB,self.labelTrain=self.load_data('data/positiveTrainNoFilter.txt','data/negativeTrainNoFilter.txt')
			self.testA,self.testB,self.labelTest=self.load_data('data/positiveTestNoFilter.txt','data/negativeTestNoFilter.txt')
		elif dataToLoad==2:
			self.trainA,self.trainB,self.labelTrain=self.get_benchmark_data('models/SuppPaddedTrain.txt')
			self.testA,self.testB,self.labelTest=self.get_benchmark_data('models/SuppPaddedTest.txt')
		elif dataToLoad==3:
			self.trainA,self.trainB,self.labelTrain=self.get_benchmark_data('models/SuppPaddedTrainCD.txt')
			self.testA,self.testB,self.labelTest=self.get_benchmark_data('models/SuppPaddedTestCD.txt')
		elif dataToLoad==4:
			self.trainA,self.trainB,self.labelTrain=self.get_benchmark_data('models/SuppPaddedTrainE.txt')
			self.testA,self.testB,self.labelTest=self.get_benchmark_data('models/SuppPaddedTestE.txt')
		elif dataToLoad==5:
			self.trainA,self.trainB,self.labelTrain=self.load_data('data/positiveTrainNoFilterPhysical.txt','data/negativeTrainNoFilterPhysical.txt')
			self.testA,self.testB,self.labelTest=self.load_data('data/positiveTestNoFilterPhysical.txt','data/negativeTestNoFilterPhysical.txt')
		elif dataToLoad==6:
			self.trainA,self.trainB,self.labelTrain=self.load_data('data/descriptors/trainNoFilterPhysicalAllPosAC.txt')
			self.testA,self.testB,self.labelTest=self.load_data('data/descriptors/testNoFilterPhysicalAllPosAC.txt')
		elif dataToLoad==7:
			self.trainA,self.trainB,self.labelTrain=self.get_benchmark_data('models/deepPPIPaddedTrain.txt')
			self.testA,self.testB,self.labelTest=self.get_benchmark_data('models/deepPPIPaddedTest.txt')
		else:
			self.trainA,self.trainB,self.labelTrain=self.get_benchmark_data('models/deepProfPPIPaddedTrain.txt')
			self.testA,self.testB,self.labelTest=self.get_benchmark_data('models/deepProfPPIPaddedTest.txt')

		# Creating the autoencoder and training it, having as labels the train data, normal procedure
		# the predict is used instead of freezing layers to create the following model, cause is more computatinal efficient
		self.train_autoencoder()
		self.model=self.create_and_train_model()
		# Testing the neural network, obtaining, the loss, accuracy, sensitivity and specficity
		self.results=self.prediction()
		# self.get_confusion_and_AUC()



	def get_benchmark_data(self, fileName):
		# Benchmark data needed some pre-processing in order to eliminate certain sequences that were outliers, 
		# due to their huge number of sequences (ex: there were sequences with 34000 amino acids)
		with open(fileName, 'r') as file:
			data1=file.read().split('\n')
			data1=data1[:len(data1)-1]
			# data1=data1[:7]			
		
		# The files loaded have 3 collumns separated by a tab, has the following example: protA protB labelOfInteraction
		labels=[int(i.split('\t')[2]) for i in data1]
		seqA=[i.split('\t')[0] for i in data1]
		seqB=[i.split('\t')[1] for i in data1]

		# seqA,seq,labels=self.remove_rare_amino(seqA,seqB,labels)
		# # Encoded seqA and seqB 
		# seqA,seqB=self.int_encode(seqA,seqB)
		
		return seqA,seqB,np.array(labels)

	def load_data(self,fileName1):
		# loads positive data
		with open(fileName1, 'r') as file:
			data1=file.read().split('\n')
			# data1=data1[:len(data1)-1]
			data1=data1[:5]
		# # loads negative data
		# with open(fileName2, 'r') as file:
		# 	data2=file.read().split('\n')
		# 	data2=data2[:len(data2)-1]
		# 	# data2=data2[:5]
		# The files loaded have 3 collumns separated by a tab, has the following example: protA protB seqA seqB labelOfInteraction

		labels1=[i.split('\t')[4] for i in data1]
		seqA1=[i.split('\t')[2] for i in data1]
		seqB1=[i.split('\t')[3] for i in data1]

		seqA,seqB=self.str_to_num(seqA1,seqB1,210)
		print(seqA)

		# Shuffle the input. Getting a random sequence of the indexes of the list
		ind=[i for i in range(0,len(labels1))]
		random.shuffle(ind)
		# Suffled labels
		labels=[int(i) for i in (labels1)]
		labels=[labels[i] for i in ind]
		A=[seqA[i] for i in ind]
		B=[seqB[i] for i in ind]
		return A,B,np.array(labels)

	def str_to_num(self,A,B,vectorLen):
		seqA=np.zeros((len(A),vectorLen), dtype='float32')
		seqB=np.zeros((len(B),vectorLen), dtype='float32')
		for i in range(len(A)):
			aa=A[i].split(',')
			bb=B[i].split(',')
			for j in range(0,len(aa)-1):
				# Integer encoding the protein if the index value does not exceed the current sequence length
				seqA[i,j]=float(aa[j])
				seqB[i,j]=float(bb[j])
		return seqA.astype('float32'),seqB.astype('float32')

	#----------------------------------------------------------------------------------------------------------------------------- 
	# LAYERS FUNCTIONS

	def input_layer(self, dtype, paddVal):
		#  the shape tuple is always defined with a hanging last dimension when the input is one-dimensional
		data=Input(shape=(paddVal,),dtype=dtype)
		return data

	def fully_connect_layer(self,neur, activationFun):
		# better performance is achieved using the ReLU activation function, which will be used in every layer, except the last one. 
		# Use a sigmoid on the output layer to ensure the network output is between 0 and 1
		layer=Dense(neur,activation=activationFun)
		return layer

	#----------------------------------------------------------------------------------------------------------------------------- 
	# CUSTOM METRICS 
	# These are special functions that are computed each epoch, so all the operations have to be according Keras functions
	# Another approach was to convert all the Keras tensor values using K.eval(y_pred) and then proceeding to make the confusion matrix
	# Essential to take in consideration that both y_true and y_pred are tensors

	def sens(self,y_true, y_pred):
		y_pred=K.round(y_pred)
		# The cast function converts the boolean output of not_equal to numbers in order to apply the sum 
		TP=K.sum(K.cast(K.not_equal(y_pred * y_true,0),'float32'))
		FN=K.sum(K.cast(K.not_equal((y_pred - 1) * y_true,0),'float32'))
		val=TP/(TP+FN)
		return val

	def spec(self,y_true, y_pred):
		y_pred=K.round(y_pred)
		# The cast function converts the boolean output of not_equal to numbers in order to apply the sum 
		TN=K.sum(K.cast(K.not_equal((y_pred - 1) * (y_true - 1),0),'float32'))
		FP=K.sum(K.cast(K.not_equal(y_pred * (y_true - 1),0),'float32'))
		val=TN/(TN+FP)
		return val

	def precision(self, y_true, y_pred):
		# quantifies the number of positive class predictions that actually are positive
		# is an approppriate metric for imbalanced classification metrics
		# Precision talks about how precise/accurate your model is out of those predicted positive
		y_pred=K.round(y_pred)
		TP=K.sum(K.cast(K.not_equal(y_pred * y_true,0),'float32'))
		FP=K.sum(K.cast(K.not_equal(y_pred * (y_true - 1),0),'float32'))
		prec_val=TP/(TP+FP)
		return prec_val

	def f1_score(self, y_true, y_pred):
		TP=K.sum(K.cast(K.not_equal(y_pred * y_true,0),'float32'))
		FP=K.sum(K.cast(K.not_equal(y_pred * (y_true - 1),0),'float32'))
		TN=K.sum(K.cast(K.not_equal((y_pred - 1) * (y_true - 1),0),'float32'))
		FN=K.sum(K.cast(K.not_equal((y_pred - 1) * y_true,0),'float32'))

		prec_val=TP/(TP+FP)
		# Applying the same understanding, we know that Recall shall be the model metric we use to 
		# select our best model when there is a high cost associated with False Negative.
		rec_val=TP/(TP+FN)
		f1=2*(prec_val*rec_val/(prec_val+rec_val))

		return f1

	def create_autoencoder(self):
		prot1=self.input_layer('float32',210)
		prot2=self.input_layer('float32',210)

		prot=concatenate([prot1,prot2])

		for i in range(0,len(self.neurons)-1):
			prot=self.fully_connect_layer(self.neurons[i],self.activationFun)(prot)

		out=self.fully_connect_layer(self.neurons[len(self.neurons)-1],self.activationFun)(prot)

		model=Model(inputs=[prot1,prot2], outputs=out)
		model.summary()

		model.compile(optimizer=self.optimizer, loss='mse', metrics=["accuracy", self.sens, self.spec, self.precision, self.f1_score])
		return model

	def train_autoencoder(self):
		print(np.shape(self.trainA))
		print(np.shape(self.trainB))
		
		autoenc=self.create_autoencoder()
		autoenc.fit(x=[self.trainA,self.trainB],y=np.concatenate((self.trainA,self.trainB),axis=1),batch_size=self.batch,epochs=self.epochs,verbose=2,callbacks=[self.earlyStop,self.checkpoint1],validation_split=0.2)

		return autoenc

	def predict_autoencoder(self,data):
		# Get the deep representations of the autoencoder
		autoenc=load_model(self.path1, custom_objects={"sens": self.sens, "spec": self.spec, "precision":self.precision, "f1_score":self.f1_score})
		autoenc=Model(inputs=autoenc.input, outputs=autoenc.layers[3].output)
		autoenc.summary()

		deepFeatures=autoenc.predict(data)

		return deepFeatures

	def create_and_train_model(self):

		# Should i standardize? minMaxScaler, robustScaler, standardScaler: https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02

		# uint8 - can represent between 0 and 255, more than enough for the 21 or 7 aminoacides
		features=self.predict_autoencoder([self.trainA,self.trainB])

		print('SHAPE: '+str(K.shape(features)))

		inputA=self.input_layer('float32',400)


		# The function requires that the output layer is configured with a single node and a ‘sigmoid‘ activation in order to predict the probability for class 1.
		# softmax function is an extension of the sigmoid function to the multiclass case
		# https://stats.stackexchange.com/questions/233658/softmax-vs-sigmoid-function-in-logistic-classifier
		# in the article they opted for sotmax instead of sigmoid
		# So the better choice for the binary classification is to use one output unit with sigmoid instead of softmax with two output units, because it will update faster
		output=self.fully_connect_layer(1, 'sigmoid')(inputA)

		model=Model(inputs=[inputA], outputs=output)
		model.summary()
		
		model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=["accuracy", self.sens, self.spec, self.precision, self.f1_score])

		values=model.fit(x=features,y=self.labelTrain,batch_size=self.batch,epochs=self.epochs,verbose=2,callbacks=[self.earlyStop,self.checkpoint2],validation_split=0.2)
		results=["{0:.2f}".format(max(i)) for i in(values.history['accuracy'], values.history['val_accuracy'])]
		self.resultsTrain=results

		return model

	# Testing with the test data
	def prediction(self):
		# The evaluate function automaticallies gives the loss, accuracy, sensitivity, and the specificity when testing, without the need to compute these metrics by myself
		# Because we use custom metrics to evaluate sensitivity and specificity we need to specify them in custom_objects
		# model=load_model(self.path, custom_objects={"sens": self.sens, "spec": self.spec, "area_under_curve":keras.metrics.AUC(name='AUC'), "precision":self.precision})
		model=load_model(self.path2, custom_objects={"sens": self.sens, "spec": self.spec, "precision":self.precision, "f1_score":self.f1_score})

		features=self.predict_autoencoder([self.testA, self.testB])

		results=["{0:.3f}".format(val) for val in model.evaluate(x=features, y=self.labelTest, verbose=0)]
		print(results)
		return results+self.resultsTrain
		# return results

	def get_confusion_and_AUC(self):
		model=load_model(self.path, custom_objects={"sens": self.sens, "spec": self.spec, "precision":self.precision, "f1_score":self.f1_score})

		y_pred=np.round(model.predict(x=[self.testA, self.testB], verbose=0))
		np.transpose(y_pred)

		auc=roc_auc_score(self.labelTest, y_pred)
		print("\nAUC: "+auc+'\n')

		# the count of true negatives is C(0,0), false negatives is C(1,0), true positives is C(1,1) and false positives is C(0,1).
		print(confusion_matrix(self.labelTest, y_pred))

		return




		




