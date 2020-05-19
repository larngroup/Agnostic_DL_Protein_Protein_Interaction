import tensorflow as tf
import random
import keras
import numpy as np
import csv
import os
from keras import *
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.callbacks import *
from keras.optimizers import *
from keras.backend import sigmoid
from keras.initializers import glorot_uniform
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects

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


class EnsembleModel:
	def __init__(self, paddVal, activationFun, neurons, batch, epochs, dropRate, 
		learnRate, path, aaSize, opt, dataToLoad):
		# Defyning the instance variables
		self.paddVal=paddVal
		# List with the number of filters for each layer, the number of elements must be equal to the number of layers
		if activationFun=='relu':
			self.activationFun='relu'
		else:
			get_custom_objects().update({'swish': Activation(self.swish)})
			self.activationFun='swish'
		self.stride=1
		self.padding='same'
		# Number of fully connected layers
		self.neurons=neurons
		self.learnRate=learnRate
		if opt==1:
		# The optimizer is Adam because is the one, that accordingly to literature is the one that gets the best results
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
		self.dataToLoad=dataToLoad
		# Callbacks
		self.earlyStop=EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=0, mode='max', baseline=None,restore_best_weights=True)
		self.checkpoint=ModelCheckpoint(filepath=self.path,monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max', period=1)
		
		if self.aaSize==7:
			self.encode={"A": 1, "R": 5, "N": 4, "D": 6, "C": 7, "E": 6, "Q": 4, "G": 1, "H": 4, "I": 2, "L": 2, "K": 5, "M": 3, "F": 2, "P": 2, "S": 3, "T": 3, "W": 4, "Y": 3, "V": 1, "U": 7}
		else:
			self.encode={"C": 1, "E": 2, "L": 3, "T": 4, "I": 5, "P": 6, "A": 7, "W": 8, "R": 9, "G": 10, "V": 11, "M": 12, "H": 13, "F": 14, "Y": 15, "Q": 16, "N": 17, "D": 18, "S": 19, "K": 20, "U": 21}
		# Loading the train and test data, shuffling it and encoding it
		if dataToLoad==1:
			self.trainA,self.trainB,self.labelTrain=self.load_data('data/positiveTrainNoFilter.txt','data/negativeTrainNoFilter.txt')
			self.testA,self.testB,self.labelTest=self.load_data('data/positiveTestNoFilter.txt','data/negativeTestNoFilter.txt')
		elif dataToLoad==2:
			self.trainA,self.trainB,self.labelTrain=self.load_benchmark_data('models/SuppPaddedTrain.txt')
			self.testA,self.testB,self.labelTest=self.load_benchmark_data('models/SuppPaddedTest.txt')
		elif dataToLoad==3:
			self.trainA,self.trainB,self.labelTrain=self.load_benchmark_data('models/SuppPaddedTrainCD.txt')
			self.testA,self.testB,self.labelTest=self.load_benchmark_data('models/SuppPaddedTestCD.txt')
		elif dataToLoad==4:
			self.trainA,self.trainB,self.labelTrain=self.load_benchmark_data('models/SuppPaddedTrainE.txt')
			self.testA,self.testB,self.labelTest=self.load_benchmark_data('models/SuppPaddedTestE.txt')
		elif dataToLoad==5:
			self.trainA,self.trainB,self.labelTrain=self.load_data('data/positiveTrainNoFilterPhysical.txt','data/negativeTrainNoFilterPhysical.txt')
			self.testA,self.testB,self.labelTest=self.load_data('data/positiveTestNoFilterPhysical.txt','data/negativeTestNoFilterPhysical.txt')
		elif dataToLoad==6:
			self.trainA1,self.trainB1,self.labelTrain=self.load_data('data/descriptors/trainNoFilterPhysicalAllPosAC.txt',210)
			self.testA1,self.testB1,self.labelTest=self.load_data('data/descriptors/testNoFilterPhysicalAllPosAC.txt',210)
			self.trainA2,self.trainB2,self.labelTrain=self.load_data('data/descriptors/trainNoFilterPhysicalAllPosCTD.txt',147)
			self.testA2,self.testB2,self.labelTest=self.load_data('data/descriptors/testNoFilterPhysicalAllPosCTD.txt',147)
			self.trainA3,self.trainB3,self.labelTrain=self.load_data('data/descriptors/trainNoFilterPhysicalAllPosConjoint.txt',343)
			self.testA3,self.testB3,self.labelTest=self.load_data('data/descriptors/testNoFilterPhysicalAllPosConjoint.txt',343)
		elif dataToLoad==7:
			self.trainA,self.trainB,self.labelTrain=self.load_benchmark_data('models/deepPPIPaddedTrain.txt')
			self.testA,self.testB,self.labelTest=self.load_benchmark_data('models/deepPPIPaddedTest.txt')
		else:
			self.trainA,self.trainB,self.labelTrain=self.load_benchmark_data('models/deepProfPPIPaddedTrain.txt')
			self.testA,self.testB,self.labelTest=self.load_benchmark_data('models/deepProfPPIPaddedTest.txt')


		#  Get the different representations
		# # self.trainA1,self.trainB1=self.ct_encode(self.trainA,self.trainB)
		# # self.testA1,self.testB1=self.ct_encode(self.testA,self.testB)
		# self.trainA2,self.trainB2=self.ac_encode(self.trainA,self.trainB)
		# self.testA2,self.testB2=self.ac_encode(self.testA,self.testB)
		# # self.trainA3,self.trainB3=self.ac_encode(self.trainA,self.trainB)
		# # self.testA3,self.testB3=self.ac_encode(self.testA,self.testB)

		# Creating the neural network model and training it
		self.model=self.train_model()
		# Testing the neural network, obtaining, the loss, accuracy, sensitivity and specficity
		self.results=self.prediction()

	def load_benchmark_data(self, fileName):
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

		seqA,seq,labels=self.remove_rare_amino(seqA,seqB,labels)
		# # Encoded seqA and seqB 
		# seqA,seqB=self.int_encode(seqA,seqB)
		
		return seqA,seqB,np.array(labels)

	def load_data(self,fileName1,vectorLen):
		# loads data
		with open(fileName1, 'r') as file:
			data1=file.read().split('\n')
			data1=data1[:len(data1)-1]
			# data1=data1[:5]

		labels1=[i.split('\t')[4] for i in data1]
		seqA1=[i.split('\t')[2] for i in data1]
		seqB1=[i.split('\t')[3] for i in data1]

		seqA,seqB=self.str_to_num(seqA1,seqB1,vectorLen)
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
	# FEATURE REPRESENTATIONS

	# def ct_encode(self,A,B):
	# 	encodedA=[]
	# 	encodedB=[]
	# 	aaProperties={'A': '1', 'G': '1', 'V': '1', 'I': '2', 'L': '2', 'F': '2', 'P': '2', 'Y': '3', 'M': '3', 'T': '3', 'S': '3', 'H': '4', 'N': '4', 'Q': '4', 'W': '4', 'R': '5', 'K': '5', 'D': '6', 'E': '6', 'C': '7'}
	# 	# Encoding with 7 aminoacid groups
	# 	for i in range(len(A)):
	# 		a=''
	# 		b=''
	# 		for j in range(len(A[i])):
	# 			a=a+aaProperties[A[i][j]]
	# 			b=b+aaProperties[B[i][j]]
	# 		encodedA.append(a)
	# 		encodedB.append(b)

	# 	# Frequency of every 3-mer
	# 	seqA=np.zeros((len(A),7*7*7), dtype='float')
	# 	seqB=np.zeros((len(B),7*7*7), dtype='float')
	# 	for i in range(len(A)):
	# 		for j in range(len(A[i])-2):
	# 			kmerA=encodedA[i][j:j+3]
	# 			indexA=int(kmerA[0]) + (int(kmerA[1])-1)*7 + (int(kmerA[2])-1)*7*7
	# 			seqA[i,indexA-1]=seqA[i,indexA-1]+1

	# 			kmerB=encodedB[i][j:j+3]
	# 			indexB=int(kmerB[0]) + (int(kmerB[1])-1)*7 + (int(kmerB[2])-1)*7*7
	# 			seqB[i,indexB-1]=seqB[i,indexB-1]+1
	# 		seqA[i,:]=seqA[i,:]/np.sum(seqA[i,:])
	# 		seqB[i,:]=seqB[i,:]/np.sum(seqB[i,:])
			
	# 	return seqA.astype('float32'),seqB.astype('float32')

	# def ac_encode(self, A,B):
	# 	# Taken from: https://github.com/smalltalkman/hppi-tensorflow/tree/master/coding
	# 	# https://www.sciencedirect.com/science/article/pii/S0025556418307168
	# 	# PCPNS: Physicochemical property names
	# 	PCPNS = ['H1', 'H2', 'NCI', 'P1', 'P2', 'SASA', 'V']

	# 	# AAPCPVS: Physicochemical property values of amino acid
	# 	AAPCPVS = {
	#     'A': { 'H1': 0.62, 'H2':-0.5, 'NCI': 0.007187, 'P1': 8.1, 'P2':0.046, 'SASA':1.181, 'V': 27.5 },
	#     'C': { 'H1': 0.29, 'H2':-1.0, 'NCI':-0.036610, 'P1': 5.5, 'P2':0.128, 'SASA':1.461, 'V': 44.6 },
	#     'D': { 'H1':-0.90, 'H2': 3.0, 'NCI':-0.023820, 'P1':13.0, 'P2':0.105, 'SASA':1.587, 'V': 40.0 },
	#     'E': { 'H1': 0.74, 'H2': 3.0, 'NCI': 0.006802, 'P1':12.3, 'P2':0.151, 'SASA':1.862, 'V': 62.0 },
	#     'F': { 'H1': 1.19, 'H2':-2.5, 'NCI': 0.037552, 'P1': 5.2, 'P2':0.290, 'SASA':2.228, 'V':115.5 },
	#     'G': { 'H1': 0.48, 'H2': 0.0, 'NCI': 0.179052, 'P1': 9.0, 'P2':0.000, 'SASA':0.881, 'V':  0.0 },
	#     'H': { 'H1':-0.40, 'H2':-0.5, 'NCI':-0.010690, 'P1':10.4, 'P2':0.230, 'SASA':2.025, 'V': 79.0 },
	#     'I': { 'H1': 1.38, 'H2':-1.8, 'NCI': 0.021631, 'P1': 5.2, 'P2':0.186, 'SASA':1.810, 'V': 93.5 },
	#     'K': { 'H1':-1.50, 'H2': 3.0, 'NCI': 0.017708, 'P1':11.3, 'P2':0.219, 'SASA':2.258, 'V':100.0 },
	#     'L': { 'H1': 1.06, 'H2':-1.8, 'NCI': 0.051672, 'P1': 4.9, 'P2':0.186, 'SASA':1.931, 'V': 93.5 },
	#     'M': { 'H1': 0.64, 'H2':-1.3, 'NCI': 0.002683, 'P1': 5.7, 'P2':0.221, 'SASA':2.034, 'V': 94.1 },
	#     'N': { 'H1':-0.78, 'H2': 2.0, 'NCI': 0.005392, 'P1':11.6, 'P2':0.134, 'SASA':1.655, 'V': 58.7 },
	#     'P': { 'H1': 0.12, 'H2': 0.0, 'NCI': 0.239531, 'P1': 8.0, 'P2':0.131, 'SASA':1.468, 'V': 41.9 },
	#     'Q': { 'H1':-0.85, 'H2': 0.2, 'NCI': 0.049211, 'P1':10.5, 'P2':0.180, 'SASA':1.932, 'V': 80.7 },
	#     'R': { 'H1':-2.53, 'H2': 3.0, 'NCI': 0.043587, 'P1':10.5, 'P2':0.291, 'SASA':2.560, 'V':105.0 },
	#     'S': { 'H1':-0.18, 'H2': 0.3, 'NCI': 0.004627, 'P1': 9.2, 'P2':0.062, 'SASA':1.298, 'V': 29.3 },
	#     'T': { 'H1':-0.05, 'H2':-0.4, 'NCI': 0.003352, 'P1': 8.6, 'P2':0.108, 'SASA':1.525, 'V': 51.3 },
	#     'V': { 'H1': 1.08, 'H2':-1.5, 'NCI': 0.057004, 'P1': 5.9, 'P2':0.140, 'SASA':1.645, 'V': 71.5 },
	#     'W': { 'H1': 0.81, 'H2':-3.4, 'NCI': 0.037977, 'P1': 5.4, 'P2':0.409, 'SASA':2.663, 'V':145.5 },
	#     'Y': { 'H1': 0.26, 'H2':-2.3, 'NCI': 117.3000, 'P1': 6.2, 'P2':0.298, 'SASA':2.368, 'V':  0.023599 },
	# 	}

	# 	def avg_sd(NUMBERS):
	# 		AVG = sum(NUMBERS)/len(NUMBERS)
	# 		TEM = [pow(NUMBER-AVG, 2) for NUMBER in NUMBERS]
	# 		DEV = sum(TEM)/len(TEM)
	# 		SD = math.sqrt(DEV)
	# 		return (AVG, SD)

	# 	# PCPVS: Physicochemical property values
	# 	PCPVS = {'H1':[], 'H2':[], 'NCI':[], 'P1':[], 'P2':[], 'SASA':[], 'V':[]}
	# 	for AA, PCPS in AAPCPVS.items():
	# 		for PCPN in PCPNS:
	# 			PCPVS[PCPN].append(PCPS[PCPN])

	# 	# PCPASDS: Physicochemical property avg and sds
	# 	PCPASDS = {}
	# 	for PCP, VS in PCPVS.items():
	# 		PCPASDS[PCP] = avg_sd(VS)

	# 	# NORMALIZED_AAPCPVS
	# 	NORMALIZED_AAPCPVS = {}
	# 	for AA, PCPS in AAPCPVS.items():
	# 		NORMALIZED_PCPVS = {}
	# 		for PCP, V in PCPS.items():
	# 			NORMALIZED_PCPVS[PCP] = (V-PCPASDS[PCP][0])/PCPASDS[PCP][1]
	# 		NORMALIZED_AAPCPVS[AA] = NORMALIZED_PCPVS

	# 	def pcp_value_of(AA, PCP):
	# 		"""Get physicochemical properties value of amino acid."""
	# 		return NORMALIZED_AAPCPVS[AA][PCP];

	# 	def pcp_sequence_of(PS, PCP):
	# 		"""Make physicochemical properties sequence of protein sequence.
	# 		"""
	# 		PCPS = []
	# 		for I, CH in enumerate(PS):
	# 			PCPS.append(pcp_value_of(CH, PCP))
	# 		# Centralization
	# 		AVG = sum(PCPS)/len(PCPS)
	# 		for I, PCP in enumerate(PCPS):
	# 			PCPS[I] = PCP - AVG
	# 		return PCPS

	# 	def ac_values_of(PS, PCP, LAG):
	# 		"""Get ac values of protein sequence."""
	# 		AVS = []
	# 		PCPS = pcp_sequence_of(PS, PCP)
	# 		for LG in range(1, LAG+1):
	# 			SUM = 0
	# 			for I in range(len(PCPS)-LG):
	# 				SUM = SUM + PCPS[I]*PCPS[I+LG]
	# 			SUM = SUM / (len(PCPS)-LG)
	# 			AVS.append(SUM)
	# 		return AVS

	# 	def all_ac_values_of(PS, LAG):
	# 		"""Get all ac values of protein sequence."""
	# 		AAVS = []
	# 		for PCP in PCPS:
	# 			AVS = ac_values_of(PS, PCP, LAG)
	# 			AAVS = AAVS + AVS
	# 		return AAVS

	# 	def ac_code_of(PS):
	# 		"""Get ac code of protein sequence."""
	# 		AC_Code = all_ac_values_of(PS, 30)
	# 		return AC_Code

	# 	# print(len(ac_code_of('MKFVYKEEHPFEKRRSEGEKIRKKYPDRVPVIVEKAPKARIGDLDKKKYLVPSDLTVGQFYFLIRKRIHLRAEDALFFFVNNVIPPTSATMGQLYQEHHEEDFFLYIAYSDESVYGL')))

	# 	# AC Values
	# 	seqA=np.zeros((len(A),30*7))
	# 	seqB=np.zeros((len(B),30*7))
	# 	for i in range(len(A)):
	# 		acA=ac_code_of(A[i])
	# 		acB=ac_code_of(B[i])
	# 		seqA[i,:]=acA
	# 		seqB[i,:]=acB

	# 	return seqA.astype('float32'),seqB.astype('float32')


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

	def input_layer(self, dtype, paddVal):
		#  the shape tuple is always defined with a hanging last dimension when the input is one-dimensional
		data=Input(shape=(paddVal,),dtype=dtype)
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

	def create_ensemble_model(self):
		# Fully connected neural network for separated features extractions

		# feat 1 - ac features
		inputA1=self.input_layer('float32',210)
		inputB1=self.input_layer('float32',210)
		# feat 2 - ct features
		inputA2=self.input_layer('float32',147)
		inputB2=self.input_layer('float32',157)
		# feat 2 - ct features
		inputA3=self.input_layer('float32',343)
		inputB3=self.input_layer('float32',343)

		neurons1=[256,128,64,32]
		neurons2=[512,256,64,32]

		feat1=concatenate([inputA1,inputB1])
		feat2=concatenate([inputA2,inputB2])
		feat3=concatenate([inputA3,inputB3])

		for i in range(0, len(neurons1)-1):
			feat1=self.fully_connect_layer(neurons1[i], self.activationFun)(feat1)
			if i!=len(neuronsFinal)-1:
				feat1=Dropout(rate=self.dropRate)(feat1)

		for i in range(0, len(neurons2)-1):
			feat2=self.fully_connect_layer(neurons2[i], self.activationFun)(feat2)
			if i!=len(neuronsFinal)-1:
				feat2=Dropout(rate=self.dropRate)(feat2)	

		for i in range(0, len(neurons2)-1):
			feat3=self.fully_connect_layer(neurons2[i], self.activationFun)(feat3)
			if i!=len(neuronsFinal)-1:
				feat3=Dropout(rate=self.dropRate)(feat3)		

		featFinal=concatenate([feat1,feat2,feat3])
		featFinal=self.fully_connect_layer(12, self.activationFun)(featFinal)
		
		output=self.fully_connect_layer(1, 'sigmoid')(featFinal)

		model=Model(inputs=[inputA1, inputB1, inputA2, inputB2, inputA3, inputB3], outputs=output)
		model.summary()
		
		# the loss function i opted with the binary crossentropy, cause is the standard one for a binary problem and
		# is a better measure than MSE for classification, because the decision boundary in a classification task is large
		model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=["accuracy", self.sens, self.spec, self.precision, self.f1_score])

		return model
		
	def train_model(self):

		bestModel=self.create_ensemble_model()
		# values=bestModel.fit(x=[self.trainA1,self.trainB1,self.trainA2,self.trainB2],y=self.labelTrain,batch_size=self.batch,epochs=self.epochs,verbose=2,callbacks=[self.earlyStop,self.checkpoint],validation_split=0.2)
		values=bestModel.fit(x=[self.trainA1,self.trainB1,self.trainA2,self.trainB2,self.trainA3,self.trainB3],y=self.labelTrain,batch_size=self.batch,epochs=self.epochs,verbose=2,callbacks=[self.earlyStop,self.checkpoint],validation_split=0.2)
		results=["{0:.2f}".format(max(i)) for i in(values.history['accuracy'], values.history['val_accuracy'])]
		self.resultsTrain=results

		return bestModel

	# Testing with the test data
	def prediction(self):
		# The evaluate function automaticallies gives the loss, accuracy, sensitivity, and the specificity when testing, without the need to compute these metrics by myself
		# Because we use custom metrics to evaluate sensitivity and specificity we need to specify them in custom_objects
		# model=load_model(self.path, custom_objects={"sens": self.sens, "spec": self.spec, "area_under_curve":keras.metrics.AUC(name='AUC'), "precision":self.precision})
		model=load_model(self.path, custom_objects={"sens": self.sens, "spec": self.spec, "precision":self.precision, "f1_score":self.f1_score})

		# results=["{0:.3f}".format(val) for val in model.evaluate(x=[self.testA1, self.testB1,self.testA2,self.testB2,self.testA2,self.testB2], y=self.labelTest, verbose=0)]
		results=["{0:.3f}".format(val) for val in model.evaluate(x=[self.testA1, self.testB1,self.testA2,self.testB2], y=self.labelTest, verbose=0)]
		print(results+self.resultsTrain)
		return results+self.resultsTrain

	def get_confusion_and_AUC(self):
		model=load_model(self.path, custom_objects={"sens": self.sens, "spec": self.spec, "precision":self.precision, "f1_score":self.f1_score})

		y_pred=np.round(model.predict(x=[self.testA1, self.testB1,self.testA2,self.testB2], verbose=0))
		np.transpose(y_pred)

		auc=roc_auc_score(self.labelTest, y_pred)
		print("\nAUC: "+auc+'\n')

		# the count of true negatives is C(0,0), false negatives is C(1,0), true positives is C(1,1) and false positives is C(0,1).
		print(confusion_matrix(self.labelTest, y_pred))

		return