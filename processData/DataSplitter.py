import numpy as np
from random import randrange
import random
# from DataCreator import DataCreator
import os

class DataSplitter:
	def __init__(self, dataCreator):
		self.dics={i[0]:i[1] for dicti in dataCreator.dics for i in dicti.items()}
		self.listDics=dataCreator.dics
		self.allPosA=dataCreator.allPosA
		self.allPosB=dataCreator.allPosB
		self.filterGO=dataCreator.filterGO
		self.interactionType=dataCreator.interactionType
		self.create_files()

	def load_dataset(self, fileName1):
		with open(fileName1, 'r') as file:
			data1=file.read().split('\n')
			data1=data1[:len(data1)-1]

		# ind=[j for j in range(len(data1)) if ("U" not in self.dics[inter.split('\t')[0]]) or ("U" not in self.dics[inter.split('\t')[1]])]
		# data1=[data1[i] for i in ind]
		
		return data1

	def write_to_txt(self,fileName,protA,protB,seqA,seqB, label):
		os.makedirs(os.path.dirname(fileName), exist_ok=True)
		with open(fileName, 'w') as file:
			for a,b,c,d in zip(protA,protB,seqA,seqB):
				file.write(str(a)+'\t'+str(b)+'\t'+str(c)+'\t'+str(d)+'\t'+str(label)+'\n')

	# only prots in the positive spplited data are added to the negative splitted data 
	def get_all_splitted_data(self,trainA,trainB, testA, testB, paddVal):
		negProtA=[]
		negProtB=[]
		numIter=0

		# It will go through the while loop twice, 
		# in the first time, it will randomly choose 2 proteins, from the ones used in train data, and if they verify all the conditions
		# the interactions will be added to the list, untill the training size is achieved. The same happens but for test data in the 2nd run

		for k in range(2):
			if self.filterGO==1:
				if k==0:
					size=len(trainA)
					A=self.GoATrain
					B=self.GoBTrain
				else:
					size=len(testA+trainA)
					A=self.GoATest
					B=self.GoBTest
			else:
				if k==0:
					size=len(trainA)
					prots=trainA+trainB
				else:
					size=len(testA+trainA)
					prots=testA+testB
			while (len(negProtA)<size):
				if self.filterGO==1:	
					i=random.randint(0,1)
					if i==0:
						sampleA=random.sample(A,1)[0]
						sampleB=random.sample(B,1)[0]
					else:
						sampleA=random.sample(B,1)[0]
						sampleB=random.sample(A,1)[0]
				else:
					sample=random.sample(prots,2)
					sampleA=sample[0]
					sampleB=sample[1]
				# Does not allow interaction with the same prot
				while(sampleA==sampleB):
					if self.filterGO==1:
						if i==0:
							sampleA=random.sample(A,1)[0]
							sampleB=random.sample(B,1)[0]
						else:
							sampleA=random.sample(B,1)[0]
							sampleB=random.sample(A,1)[0]
					else:
						sample=random.sample(prots,2)
						sampleA=sample[0]
						sampleB=sample[1]

				state=True
				# Verifyes if the interaction already exsits in all the positive data
				for a, b in zip(self.allPosA,self.allPosB):
					if a==sampleA and b==sampleB:
						state=False
						break
				# Verifyes if the inverse interaction already exsits in positive data
					elif a==sampleB and b==sampleA:
						state=False
						break
				# Verifyes if the interaction already exsits in negative data
				if state==True:
					for c,d in zip(negProtA, negProtB):
						if sampleA==c and sampleB==d:
							state=False
							break
				# Verifyes if the inverse interaction already exsits in negative data
						elif sampleA==d and sampleB==c:
							state=False
							break
					if state==True and len(self.dics[sampleA])<paddVal and len(self.dics[sampleB])<paddVal:
						negProtA.append(sampleA)
						negProtB.append(sampleB)
		# print(negProtA,negProtB)
		# Get the sequences
		seqNegA=[self.dics[i] for i in negProtA]
		seqNegB=[self.dics[i] for i in negProtB]
		trainAPos=[self.dics[i] for i in trainA]
		trainBPos=[self.dics[i] for i in trainB]
		testAPos=[self.dics[i] for i in testA]
		testBPos=[self.dics[i] for i in testB]

		return negProtA[:len(trainA)], negProtB[:len(trainA)],seqNegA[:len(trainA)], seqNegB[:len(trainA)], negProtA[len(trainA):], negProtB[len(trainA):],seqNegA[len(trainA):], seqNegB[len(trainA):],trainAPos,trainBPos,testAPos,testBPos			

	# Splits the positive data in train and test, with a specific ratio
	def split_positive_data(self,seqA,seqB,ratio,size):


		index=[]
		while len(index)<int(size*ratio):
			i=random.randint(0,int(len(seqA))-1)
			if i not in index:index.append(i)
		trainSeqA=[seqA[i] for i in index]
		trainSeqB=[seqB[i] for i in index]

		indexTest=[]
		while len(indexTest)<int(size-size*ratio):
			i=random.randint(0,int(len(seqA))-1)
			if i not in index and i not in indexTest:indexTest.append(i)
		testSeqA=[seqA[i] for i in indexTest]
		testSeqB=[seqB[i] for i in indexTest]

		# Gets the proteins only used in train for both GOs
		# To posteriorly guarantee that the negative train data contains the same proteins as the positive train data
		if self.filterGO==1:
			GoATrain=[]
			GoBTrain=[]
			for a,b in zip(trainSeqA, trainSeqB):
				if a in self.listDics[0]:
					GoATrain.append(a)
					GoBTrain.append(b)
				elif a in self.listDics[1]:
					GoATrain.append(b)
					GoBTrain.append(a)

			# Gets the proteins only used in test for both GOs
			GoATest=[]
			GoBTest=[]
			for a,b in zip(testSeqA, testSeqB):
				if a in self.listDics[0]:
					GoATest.append(a)
					GoBTest.append(b)
				elif a in self.listDics[1]:
					GoATest.append(b)
					GoBTest.append(a)

			self.GoATrain=GoATrain
			self.GoBTrain=GoBTrain
			self.GoATest=GoATest
			self.GoBTest=GoBTest

		return trainSeqA,trainSeqB,testSeqA,testSeqB

	def padding(self,data, maxLen):
		print("Pre-padding len: ",len(data))
		# Delete interactions with protein sequences that exceed maxLen
		data=[inter for inter in data if all(len(prot)<=maxLen for prot in (self.dics[inter.split('\t')[0]],self.dics[inter.split('\t')[1]]))]
		print("Pos-padding len: ",len(data))

		seqA=[i.split('\t')[0] for i in data]
		seqB=[i.split('\t')[1] for i in data]

		return seqA,seqB

	def get_padding_value(self,pos):
		lens=[len(self.dics[inter.split('\t')[i]]) for inter in pos for i in range(0,2)]
		maxLen=int(np.percentile(lens,90))
		print("\nPadding value: ",maxLen)
		return maxLen

	def create_files(self):
		if self.interactionType==1:
			if self.filterGO==1:
				pos=self.load_dataset('data/positiveDataFiltered.txt')
			else:
				pos=self.load_dataset('data/positiveDataNoFilter.txt')
		else:
			if self.filterGO==1:
				pos=self.load_dataset('data/positiveDataFilteredPhysical.txt')
			else:
				# needs to be changed if is for allpos or not
				pos=self.load_dataset('data/positiveDataNoFilterPhysical.txt')


		maxLen=self.get_padding_value(pos)
		print('\nPositive Data:')
		posA,posB=self.padding(pos, maxLen)
		posTrainA,posTrainB,posTestA,posTestB=self.split_positive_data(posA, posB, 0.7, len(posA))
		# print(posTrainA,posTrainB,posTestA,posTestB)
		print('\nNegative Data:')
		protTrainA,protTrainB,negTrainA,negTrainB,protTestA,protTestB,negTestA,negTestB,seqPTrainA,seqPTrainB,seqPTestA,seqPTestB=self.get_all_splitted_data(posTrainA,posTrainB,posTestA,posTestB,maxLen)
		print('POS TRAIN SIZE: ',len(posTrainA),'NEG TRAIN SIZE: ',len(negTrainA))
		print('POS TEST SIZE: ',len(posTestA),'NEG TEST SIZE: ',len(negTestA))
		if self.interactionType==1:
			if self.filterGO==1:
				self.write_to_txt("data/positiveTrainFiltered.txt",posTrainA,posTrainB,seqPTrainA,seqPTrainB,1)
				self.write_to_txt("data/positiveTestFiltered.txt",posTestA,posTestB,seqPTestA,seqPTestB,1)
				self.write_to_txt("data/negativeTrainFiltered.txt",protTrainA,protTrainB,negTrainA,negTrainB,0)
				self.write_to_txt("data/negativeTestFiltered.txt",protTestA,protTestB,negTestA,negTestB,0)
			else:
				self.write_to_txt("data/positiveTrainNoFilter.txt",posTrainA,posTrainB,seqPTrainA,seqPTrainB,1)
				self.write_to_txt("data/positiveTestNoFilter.txt",posTestA,posTestB,seqPTestA,seqPTestB,1)
				self.write_to_txt("data/negativeTrainNoFilter.txt",protTrainA,protTrainB,negTrainA,negTrainB,0)
				self.write_to_txt("data/negativeTestNoFilter.txt",protTestA,protTestB,negTestA,negTestB,0)
		else:
			if self.filterGO==1:
				self.write_to_txt("data/positiveTrainFilteredPhysical.txt",posTrainA,posTrainB,seqPTrainA,seqPTrainB,1)
				self.write_to_txt("data/positiveTestFilteredPhysical.txt",posTestA,posTestB,seqPTestA,seqPTestB,1)
				self.write_to_txt("data/negativeTrainFilteredPhysical.txt",protTrainA,protTrainB,negTrainA,negTrainB,0)
				self.write_to_txt("data/negativeTestFilteredPhysical.txt",protTestA,protTestB,negTestA,negTestB,0)
			else:
				self.write_to_txt("data/positiveTrainNoFilterPhysical.txt",posTrainA,posTrainB,seqPTrainA,seqPTrainB,1)
				self.write_to_txt("data/positiveTestNoFilterPhysical.txt",posTestA,posTestB,seqPTestA,seqPTestB,1)
				self.write_to_txt("data/negativeTrainNoFilterPhysical.txt",protTrainA,protTrainB,negTrainA,negTrainB,0)
				self.write_to_txt("data/negativeTestNoFilterPhysical.txt",protTestA,protTestB,negTestA,negTestB,0)

		
		# self.write_to_txt("data/positiveTrain.txt",posTrainA,posTrainB,1)
		# self.write_to_txt("data/positiveTest.txt",posTestA,posTestB,1)
		# self.write_to_txt("data/negativeTrain.txt",negTrainA,negTrainB,0)
		# self.write_to_txt("data/negativeTest.txt",negTestA,negTestB,0)

# a=DataCreator(['chromatinbinding0003682.fasta','organiccycliccompoundbinding0097159.fasta'])
# b=DataSplitter(a)
