# # os.environ["CUDA_VISIBLE_DEVICES"]="0"
# # config = tf.compat.v1.ConfigProto()
# # #  Allowing growth handles not to allocate all the memory.
# # config.gpu_options.allow_growth = True
# # session = tf.compat.v1.Session(config=config)
# # tf.compat.v1.keras.backend.set_session(session)
# # K.set_session(session)
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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
# from keras.layers import Activation

# If a TensorFlow operation has both CPU and GPU implementations, 
# TensorFlow will automatically place the operation to run on a GPU device first


#----------------------------------------------------------------------------------------------------------------------------- 
	# CUSTOM ACTIVATION FUNCTION
	# SWISH
	# https://www.bignerdranch.com/blog/implementing-swish-activation-function-in-keras/
	
	# ------------------------------------------------------------
	# needs to be defined as activation class otherwise error
	# AttributeError: 'Activation' object has no attribute '__name__'  
	# https://github.com/keras-team/keras/issues/8716  
class Swish(Activation):
	    
	def __init__(self, activation, **kwargs):
	        super(Swish, self).__init__(activation, **kwargs)
	        self.__name__ = 'swish'

def swish(x):
	return (K.sigmoid(x) * x)

	
class CnnModel:

	def __init__(self, paddVal, activationFun, numConvLayers, numFilters, kernelSize, poolSize, numFcLayers,
	 neurons, batch, epochs, dropRate, learnRate, path, aaSize, architecture_num,cv, opt, dataToLoad, concatenateOrMultiply):
		# Defyning the instance variables
		self.concatenateOrMultiply=concatenateOrMultiply
		# my-1367 or teacher-1267
		self.paddVal=paddVal
		# Number of convolutional layers
		self.numConvLayers=numConvLayers
		# List with the number of filters for each layer, the number of elements must be equal to the number of layers
		self.numFilters=numFilters
		self.kernelSize=kernelSize
		if activationFun=='relu':
			self.activationFun='relu'
		else:
			get_custom_objects().update({'swish':Swish(swish)})
			self.activationFun='swish'
		self.poolSize=poolSize
		self.stride=1
		self.padding='same'
		# Number of fully connected layers
		self.numFcLayers=numFcLayers
		self.neurons=neurons
		self.learnRate=learnRate
		if opt==1:
		# The optimizer is Adam because, accordingly to literature is the one that gets the best results
			self.optimizer=Adam(lr=learnRate)
		elif opt==2:
			self.optimizer=SGD(lr=learnRate)
		else:
			self.optimizer=RMSprop(lr=learnRate)
		self.batch=batch
		self.epochs=epochs
		self.dropRate=dropRate
		# Path to save the model architecture and the weights in a .h5 file
		self.path=path
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
		self.architecture_num=architecture_num
		self.dataToLoad=dataToLoad
		# Callbacks
		self.earlyStop=EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=0, mode='max', baseline=None,restore_best_weights=True)
		self.checkpoint=ModelCheckpoint(filepath=self.path,monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
		# cv fold number 
		self.cv=cv
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
			self.trainA,self.trainB,self.labelTrain=self.load_data('data/positiveTrainNoFilterPhysicalAllPos.txt','data/negativeTrainNoFilterPhysicalAllPos.txt')
			self.testA,self.testB,self.labelTest=self.load_data('data/positiveTestNoFilterPhysicalAllPos.txt','data/negativeTestNoFilterPhysicalAllPos.txt')
		elif dataToLoad==7:
			self.trainA,self.trainB,self.labelTrain=self.get_benchmark_data('models/deepPPIPaddedTrain.txt')
			self.testA,self.testB,self.labelTest=self.get_benchmark_data('models/deepPPIPaddedTest.txt')
		else:
			self.trainA,self.trainB,self.labelTrain=self.get_benchmark_data('models/deepProfPPIPaddedTrain.txt')
			self.testA,self.testB,self.labelTest=self.get_benchmark_data('models/deepProfPPIPaddedTest.txt')

		# Creating the neural network model and training it
		self.model=self.train_model()
		# Testing the neural network, obtaining, the loss, accuracy, sensitivity and specficity
		self.results=self.prediction()
		# self.get_confusion_and_AUC()



	def get_benchmark_data(self, fileName):
		# Benchmark data needed some pre-processing in order to eliminate certain sequences that were outliers, 
		# due to their huge number of sequences (ex: there were sequences with 34000 amino acids)
		with open(fileName, 'r') as file:
			data1=file.read().split('\n')
			data1=data1[:len(data1)-1]
		
		# The files loaded have 3 collumns separated by a tab, has the following example: protA protB labelOfInteraction
		labels=[i.split('\t')[2] for i in data1]
		seqA=[i.split('\t')[0] for i in data1]
		seqB=[i.split('\t')[1] for i in data1]

		# Encoded seqA and seqB 
		seqA,seqB=self.int_encode(seqA,seqB)

		return seqA,seqB,np.array(labels)


	def load_data(self,fileName1, fileName2):
		# loads positive data
		with open(fileName1, 'r') as file:
			data1=file.read().split('\n')
			data1=data1[:len(data1)-1]
			# data1=data1[:10]
		# loads negative data
		with open(fileName2, 'r') as file:
			data2=file.read().split('\n')
			data2=data2[:len(data2)-1]
			# data2=data2[:10]
		# The files loaded have 3 collumns separated by a tab, has the following example: protA protB seqA seqB labelOfInteraction
		labels1=[i.split('\t')[4] for i in data1]
		labels2=[i.split('\t')[4] for i in data2]
		seqA1=[i.split('\t')[2] for i in data1]
		seqA2=[i.split('\t')[2] for i in data2]
		seqB1=[i.split('\t')[3] for i in data1]
		seqB2=[i.split('\t')[3] for i in data2]

		# Shuffle the input. Getting a random sequence of the indexes of the list
		ind=[i for i in range(0,len(labels1+labels2))]
		random.shuffle(ind)
		# Suffled labels
		# when data is too big use tf.data dataset
		labels=np.array([int(i) for i in (labels1+labels2)])[ind]
		# Encoded seqA and seqB 
		seqA,seqB=self.int_encode(seqA1+seqA2,seqB1+seqB2)
		# returns the shuffled seqA and seqB 
		return seqA[ind,:],seqB[ind,:],labels

	def int_encode(self,A,B):
		seqA=np.zeros((len(A),self.paddVal), dtype='uint8')
		seqB=np.zeros((len(B),self.paddVal), dtype='uint8')
		for i in range(len(A)):
			for j in range(self.paddVal):
				# Integer encoding the protein if the index value does not exceed the current sequence length
				if j<len(A[i]):
					seqA[i,j]=self.encode[A[i][j]]
				if j<len(B[i]):
					seqB[i,j]=self.encode[B[i][j]]
		return seqA.astype('uint8'),seqB.astype('uint8')

	#----------------------------------------------------------------------------------------------------------------------------- 
	# LAYERS FUNCTIONS

	def one_hot_layer(self):
		# using keras.utils.to_categorical during preprocessing can take too long and may make the dataset file bigger
		# So i implemented a lambda layer, that allows to apply a custom function to the data
		# That custom function is the K.one_hot
		# Firstly cast x in an integer tensor, next pass it to K.one_hot along with num_classes (the length of the one-hot vector)
	    def one_hot(x, num_classes):
	        return K.one_hot(K.cast(x, 'uint8'),num_classes=num_classes)
	    return Lambda(one_hot, arguments={'num_classes': self.aaSize}, input_shape=(self.paddVal,))

	def input_layer(self, dtype):
		#  the shape tuple is always defined with a hanging last dimension when the input is one-dimensional
		data=Input(shape=(self.paddVal,),dtype=dtype)
		return data

	def conv_layer(self,numFilt):
		layer=Conv1D(filters=numFilt,kernel_size=self.kernelSize,strides=self.stride,padding=self.padding,activation=self.activationFun)
		return layer

	def pooling_layer(self):
		layer=MaxPooling1D(pool_size=self.poolSize,padding=self.padding)
		return layer

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


	#----------------------------------------------------------------------------------------------------------------------------- 
	# MODEL ARCHITECTURE 
	# During this work i considered 4 different architectures, note that n is the number of caonvolutional layers,
	# and m the number of fully connected, both determined by the user
	# 
	# 1) Global:
	#		[Convolutional]^n -> GlobalMaxPooling ->[FC]^m -> OutputLayer
	# 
	# 		to get this the poolingType='global', fcOrGap='fc' 
	# 
	# 2) Gap:
	#		[Convolutional -> MaxPooling]^n -> GlobalAveragePooling -> OutputLayer
	# 
	# 		to get this the poolingType='local', fcOrGap='gap'
	# 
	# 3) Local (the standard approach):
	#		[Convolutional -> MaxPooling]^n -> [FC]^m -> OutputLayer
	# 
	# 		to get this the poolingType='local', fcOrGap='fc' 
	# 
	# 4) ResNet

	def architecture_1(self,a, b):
	# 1) Global:
	# [Convolutional]^n -> GlobalMaxPooling ->[FC]^m -> OutputLayer

		for i in range(0,self.numConvLayers):
			a=self.conv_layer(self.numFilters[i])(a)
			b=self.conv_layer(self.numFilters[i])(b)

		poolA=GlobalMaxPooling1D()(a)
		poolB=GlobalMaxPooling1D()(b)
		if self.concatenateOrMultiply=='concatenate':
			prot=concatenate([poolA,poolB])
		else:
			prot=multiply([poolA,poolB])

		for i in range(0, self.numFcLayers):
			prot=self.fully_connect_layer(self.neurons[i], self.activationFun)(prot)
			# Condition to not apply dropout after the last dense layer, because the network, won´t have time to take use of the dropout
			# as the following layer will be the classification layer 
			if i!=self.numFcLayers-1:
				prot=Dropout(rate=self.dropRate)(prot)

		return prot

	def architecture_2(self,a,b):
		# 2) Gap:
		# [Convolutional -> MaxPooling]^n -> GlobalAveragePooling -> OutputLayer

		for i in range(0,self.numConvLayers):
			a=self.conv_layer(self.numFilters[i])(a)
			b=self.conv_layer(self.numFilters[i])(b)
			a=self.pooling_layer()(a)
			b=self.pooling_layer()(b)

		poolA=GlobalAveragePooling1D()(a)
		poolB=GlobalAveragePooling1D()(b)
		if self.concatenateOrMultiply=='concatenate':
			prot=concatenate([poolA,poolB])
		else:
			prot=multiply([poolA,poolB])

		return prot

	def architecture_3(self,a,b):
		# 3) Local (the standard approach):
		# [Convolutional -> MaxPooling]^n -> [FC]^m -> OutputLayer

		for i in range(0,self.numConvLayers):
			a=self.conv_layer(self.numFilters[i])(a)
			b=self.conv_layer(self.numFilters[i])(b)
			a=self.pooling_layer()(a)
			b=self.pooling_layer()(b)

		poolA=Flatten()(a)
		poolB=Flatten()(b)
		if self.concatenateOrMultiply=='concatenate':
			prot=concatenate([poolA,poolB])
		else:
			prot=multiply([poolA,poolB])

		for i in range(0, self.numFcLayers):
			prot=self.fully_connect_layer(self.neurons[i], self.activationFun)(prot)
			# Condition to not apply dropout after the last dense layer, because the network, won´t have time to take use of the dropout
			# as the following layer will be the classification layer 
			if i!=self.numFcLayers-1:
				prot=Dropout(rate=self.dropRate)(prot)

		return prot

	# https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
	def identity_block(self,X, f, filters):
		# Retrieve Filters
		F1, F2, F3 = filters
		
		# Save the input value. Needed later to add back  
		X_initial = X
		X=Conv1D(filters = F1, kernel_size = 1, strides = 1, padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
		X=BatchNormalization()(X)
		X=Activation('relu')(X)
		X=Conv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
		X=BatchNormalization()(X)
		X=Activation('relu')(X)
		X=Conv1D(filters = F3, kernel_size = 1, strides = 1, padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
		X=BatchNormalization()(X)
		# Final step: Add shortcut value to main path, and pass it through a RELU activation
		X=Add()([X, X_initial])
		X=Activation('relu')(X)

		return X

	def convolutional_block(self,inputX, f, filters, s = 2):
		F1, F2, F3 = filters
		# Save the input value. Needed later to add back  
		X_initial = inputX
		# Main Path 
		X=Conv1D(F1, kernel_size = 1, strides = s, kernel_initializer = glorot_uniform(seed=0))(inputX)
		X=BatchNormalization()(X)
		X=Activation('relu')(X)

		X=Conv1D(filters = F2, kernel_size = f, strides = 1, padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
		X=BatchNormalization()(X)
		X=Activation('relu')(X)
		X=Conv1D(filters = F3, kernel_size = 1, strides = 1, padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
		X=BatchNormalization()(X)
		# Shortcut path
		X_initial=Conv1D(filters = F3, kernel_size = 1, strides = s, padding = 'valid',kernel_initializer = glorot_uniform(seed=0))(X_initial)
		X_initial=BatchNormalization()(X_initial)

		# Final step: Add shortcut value to main path
		X=Add()([X, X_initial])
		X=Activation('relu')(X)
		return X

	def architecture_4(self,a,b):
		# Protein A
		a=Conv1D(64, 7, strides=2, kernel_initializer=glorot_uniform(seed=0))(a)
		a=BatchNormalization()(a)
		a=Activation('relu')(a)
		a=MaxPooling1D(3, strides=2)(a)

		a=self.convolutional_block(a, f=3, filters=[64,64,256])
		a=self.identity_block(a, 3, [64, 64, 256])
		a=self.identity_block(a, 3, [64, 64, 256])

		a=self.convolutional_block(a, f=3, filters=[128, 128, 512])
		a=self.identity_block(a, 3, [128, 128, 512])
		a=self.identity_block(a, 3, [128, 128, 512])
		a=self.identity_block(a, 3, [128, 128, 512])

		a=self.convolutional_block(a, f=3, filters=[256, 256, 1024])
		a=self.identity_block(a, 3, [256, 256, 1024])
		a=self.identity_block(a, 3, [256, 256, 1024])
		a=self.identity_block(a, 3, [256, 256, 1024])
		# a=self.identity_block(a, 3, [256, 256, 1024])
		# a=self.identity_block(a, 3, [256, 256, 1024])

		# a=self.convolutional_block(a, f=3, filters=[512, 512, 2048])
		# a=self.identity_block(a, 3, [512, 512, 2048])
		# a=self.identity_block(a, 3, [512, 512, 2048])
		a=AveragePooling1D(2)(a)

		# Protein B
		b=Conv1D(64, 7, strides=2, kernel_initializer=glorot_uniform(seed=0))(b)
		b=BatchNormalization()(b)
		b=Activation('relu')(b)
		b=MaxPooling1D(3, strides=2)(b)

		b=self.convolutional_block(b, f=3, filters=[64,64,256])
		b=self.identity_block(b, 3, [64,64,256])
		b=self.identity_block(b, 3, [64,64,256])

		b=self.convolutional_block(b, f=3, filters=[128,128,512])
		b=self.identity_block(b, 3, [128,128,512])
		b=self.identity_block(b, 3, [128,128,512])
		b=self.identity_block(b, 3, [128,128,512])

		b=self.convolutional_block(b, f=3, filters=[256,256,1024])
		b=self.identity_block(b, 3, [256,256,1024])
		b=self.identity_block(b, 3, [256,256,1024])
		b=self.identity_block(b, 3, [256,256,1024])
		# b=self.identity_block(b, 3, [256,256,1024])
		# b=self.identity_block(b, 3, [256,256,1024])

		# b=self.convolutional_block(b, f=3, filters=[512,512,2048])
		# b=self.identity_block(b, 3, [512,512,2048])
		# b=self.identity_block(b, 3, [512,512,2048])
		b=AveragePooling1D(2)(b)

		poolA=Flatten()(a)
		poolB=Flatten()(b)
		prot=concatenate([poolA,poolB])

		return prot

	def create_model(self):

		# uint8 - can represent between 0 and 255, more than enough for the 21 or 7 aminoacides
		inputA=self.input_layer('uint8')
		inputB=self.input_layer('uint8')

		a=self.one_hot_layer()(inputA)
		b=self.one_hot_layer()(inputB)

		# a=self.ct_coding_layer()(inputA)
		# b=self.ct_coding_layer()(inputB)

		architectures=[self.architecture_1, self.architecture_2, self.architecture_3, self.architecture_4]

		prot=architectures[self.architecture_num-1](a,b)

		# The function requires that the output layer is configured with a single node and a ‘sigmoid‘ activation in order to predict the probability for class 1.
		output=self.fully_connect_layer(1, 'sigmoid')(prot)

		model=Model(inputs=[inputA, inputB], outputs=output)
		model.summary()
		
		# the loss function i opted with the binary crossentropy, cause is the standard one for a binary problem and
		# is a better measure than MSE for classification, because the decision boundary in a classification task is large
		# model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=["accuracy", self.sens, self.spec, keras.metrics.AUC(name='AUC'), self.precision])
		model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=["accuracy", self.sens, self.spec, self.precision, self.f1_score])

		return model

	def train_model(self):

		if self.cv==1:
			bestModel=self.create_model()
			values=bestModel.fit(x=[self.trainA,self.trainB],y=self.labelTrain,batch_size=self.batch,epochs=self.epochs,verbose=2,callbacks=[self.earlyStop,self.checkpoint],validation_split=0.2)
			results=["{0:.2f}".format(max(i)) for i in(values.history['accuracy'], values.history['val_accuracy'])]
			self.resultsTrain=results
		else:
			bestAcc=0
			bestModel=None
			# The stratification is done based on the labels
			folds = list(StratifiedKFold(n_splits=self.cv, shuffle=True).split(self.trainA, self.labelTrain))
			model=self.create_model()
			for fold, (i, j) in enumerate(folds):
				print('Fold: '+ str(fold))
				values=model.fit(x=[self.trainA[i],self.trainB[i]],y=self.labelTrain[i],batch_size=self.batch,epochs=self.epochs,verbose=2,callbacks=[self.earlyStop,self.checkpoint],validation_data=([self.trainA[j],self.trainB[j]], self.labelTrain[j]))
				if float(max(values.history['val_accuracy']))>bestAcc:
					bestAcc=float(max(values.history['val_accuracy']))
					bestModel=model
					results=["{0:.3f}".format(max(i)) for i in(values.history['accuracy'], values.history['val_accuracy'])]
					self.resultsTrain=results
			# At the end the bestModel is resaved in the path that has been saving the other models trained in the cv
			bestModel.save(self.path)

		return bestModel

	# Testing with the test data
	def prediction(self):
		# The evaluate function automaticallies gives the loss, accuracy, sensitivity, and the specificity when testing, without the need to compute these metrics by myself
		# Because we use custom metrics to evaluate sensitivity and specificity we need to specify them in custom_objects
		# model=load_model(self.path, custom_objects={"sens": self.sens, "spec": self.spec, "area_under_curve":keras.metrics.AUC(name='AUC'), "precision":self.precision})
		model=load_model(self.path, custom_objects={"sens": self.sens, "spec": self.spec, "precision":self.precision, "f1_score":self.f1_score})

		results=["{0:.3f}".format(val) for val in model.evaluate(x=[self.testA, self.testB], y=self.labelTest, verbose=0)]
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
		




