import numpy as np
import random
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

class Svm:

	def __init__(self, aaSize, paddVal,inputType):
		self.paramSVM=[{"kernel": ["linear"],"C": [1, 10, 100]},{"kernel": ["rbf"],"C": [1, 10],"gamma": [1e-2, 1e-3]}]

		self.paddVal=paddVal
		self.aaSize=aaSize
		if self.aaSize==7:
			self.encode={"A": 1, "R": 5, "N": 4, "D": 6, "C": 7, "E": 6, "Q": 4, "G": 1, "H": 4, "I": 2, "L": 2, "K": 5, "M": 3, "F": 2, "P": 2, "S": 3, "T": 3, "W": 4, "Y": 3, "V": 1, "U": 7}
		else:
			self.encode={"C": 1, "E": 2, "L": 3, "T": 4, "I": 5, "P": 6, "A": 7, "W": 8, "R": 9, "G": 10, "V": 11, "M": 12, "H": 13, "F": 14, "Y": 15, "Q": 16, "N": 17, "D": 18, "S": 19, "K": 20, "U": 21}

		self.trainA,self.trainB,self.labelTrain=self.load_data('../data/positiveTrain.txt','../data/negativeTrain.txt')
		self.testA,self.testB,self.labelTest=self.load_data('../data/positiveTest.txt','../data/negativeTest.txt')
		self.inputType=inputType
		self.inputTrain=self.create_input(self.trainA,self.trainB)
		self.inputTest=self.create_input(self.testA,self.testB)
		self.model=self.train_svm()
		self.testing_best_model()




	def load_data(self,fileName1, fileName2):
		# loads positive data
		with open(fileName1, 'r') as file:
			data1=file.read().split('\n')
			data1=data1[:len(data1)-1]
			# data1=data1[:100]
		# loads negative data
		with open(fileName2, 'r') as file:
			data2=file.read().split('\n')
			data2=data2[:len(data2)-1]
			# data2=data2[:100]
		# The files loaded have 3 collumns separated by a tab, has the following example: protA protB labelOfInteraction
		labels1=[i.split('\t')[2] for i in data1]
		labels2=[i.split('\t')[2] for i in data2]
		seqA1=[i.split('\t')[0] for i in data1]
		seqA2=[i.split('\t')[0] for i in data2]
		seqB1=[i.split('\t')[1] for i in data1]
		seqB2=[i.split('\t')[1] for i in data2]

		# Shuffle the input. Getting a random sequence of the indexes of the list
		ind=[i for i in range(0,len(labels1+labels2))]
		random.shuffle(ind)
		# Suffled labels
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

	def get_frequencies_vector(self,A,B):
		freqsA=np.zeros((self.aaSize,))
		freqsB=np.zeros((self.aaSize,))
		for i in range(len(A)):
			freqsA[A[i]-1]=freqsA[A[i]-1]+1

		for i in range(len(B)):
			freqsB[B[i]-1]=freqsB[B[i]-1]+1
		
		freqsA=freqsA/len(A)
		freqsB=freqsA/len(B)

		return np.concatenate((freqsA,freqsB))

	def get_consecutive_3_frequencies(self,A,B):
		dicA={}
		dicB={}
		for i in range(1,self.aaSize+1):
			for j in range(1,self.aaSize+1):
				for k in range(1,self.aaSize+1):
					keyDic=str(i)+','+str(j)+','+str(k)
					dicA[keyDic]=0
					dicB[keyDic]=0

		for i in range(len(A)-2):
			keyDic=str(A[i])+','+str(A[i+1])+','+str(A[i+2])
			dicA[keyDic]+=1

		for i in range(len(B)-2):
			keyDic=str(B[i])+','+str(B[i+1])+','+str(B[i+2])
			dicB[keyDic]+=1
		
		freqsA=[dicA[x]/len(A) for x in dicA]
		freqsB=[dicB[x]/len(B) for x in dicB]
		return freqsA+freqsB

	def create_input(self, A, B):
		inputt=[]
		for i in range(len(A)):
			if self.inputType=='freq1':
				vector=self.get_frequencies_vector(A[i],B[i])
			else:
				vector=self.get_consecutive_3_frequencies(A[i],B[i])
			inputt.append(vector)
		print(np.array(inputt).shape)
		return np.array(inputt)
		# return inputt

	def train_svm(self):
		 # take as input two arrays: an array X of size [n_samples, n_features]
		 # holding the training samples, and an array y of class labels (strings or integers), size [n_samples]:
		model=svm.SVC()
		 # cv=None defalt 5-fold cv
		# model.fit(self.inputTrain, self.labelTrain)
		grid=GridSearchCV(model, self.paramSVM, verbose=1)
		grid.fit(self.inputTrain, self.labelTrain)

		bestModel=grid.best_estimator_
		print('Best parameters: ',grid.best_params_)

		means=grid.cv_results_['mean_test_score']
		for mean, params in zip(means, grid.cv_results_['params']):
			print("%0.3f accuracy - with params: %r" % (mean, params))

		# save model
		filename = 'SvmModel/best_SVM_model_.sav'
		joblib.dump(bestModel, filename) 

		return bestModel
		# return model

	def testing_best_model(self):
		y_true,y_pred=self.labelTest,self.model.predict(self.inputTest)
		print(confusion_matrix(y_true,y_pred))
		tn,fp,fn,tp=confusion_matrix(y_true,y_pred).ravel()
		print(tn,fp,fn,tp)
		print("accuracy: %0.3f , sens: %0.3f" % (((tn+tp)/(tn+fp+fn+tp)), (tp/(tp+fn))))


# def get_consecutive_3(A,dicc):
# 		dicA={}
# 		dicB={}
# 		for i in range(1,dicc+1):
# 			for j in range(1,dicc+1):
# 				for k in range(1,dicc+1):
# 					keyDic=str(i)+','+str(j)+','+str(k)
# 					dicA[keyDic]=0
# 		print(dicA)
# 		for i in range(len(A)-2):
# 			keyDic=str(A[i])+','+str(A[i+1])+','+str(A[i+2])
# 			if keyDic in dicA:
# 				dicA[keyDic]+=1
# 			else:
# 				dicA[keyDic]=1
# 		print(dicA)
# 		# for i in range(len(B)-2):
# 		# 	keyDic=str(B[i])+','+str(B[i+1])+','+str(B[i+2])
# 		# 	if keyDic in dicB:
# 		# 		dicB[keyDic]+=1
# 		# 	else:
# 		# 		dicB[keyDic]=1
		
# 		freqsA=[dicA[x]/len(A) for x in dicA]
# 		# freqsB=[x/len(B) for x in freqsB]
# 		return freqsA,dicA

# A=[1,2,3,1,2,3,1,2,3,4]
# print(get_consecutive_3(A,4))

# def get_frequencies_vector(A,B):
# 	freqsA=np.zeros((5,))
# 	freqsB=np.zeros((5,))
# 	for i in range(len(A)):
# 		freqsA[A[i]-1]=freqsA[A[i]-1]+1

# 	for i in range(len(B)):
# 		freqsB[B[i]-1]=freqsB[B[i]-1]+1
	
# 	freqsA=freqsA/len(A)
# 	freqsB=freqsA/len(B)
# 	print(freqsA)
# 	return np.concatenate((freqsA,freqsB))
# A=[1,2,3,1,2,3,1,2,3,4]
# print(get_frequencies_vector(A,[]))
a=Svm(7,1367,'freqs3')
# X = np.array([np.array([-1, -1]), np.array([-2, -1]), np.array([1, 1]), np.array([2, 1])])
# y = np.array([1, 1, 2, 2])
# clf = svm.SVC(gamma='auto')
# clf.fit(X, y)
# print(clf.predict([[-0.8, -1]]))
