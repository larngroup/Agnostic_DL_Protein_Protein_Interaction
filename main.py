from models.CnnModel import CnnModel
from models.EnsembleModel import EnsembleModel
from models.ConvLstm import ConvLstm
from models.FullyCnnModel import FullyCnnModel
from models.AutoEncoderModel import AutoEncoderModel
from processData.DataCreator import DataCreator
from processData.DataSplitter import DataSplitter
import random
import csv
import numpy as np
import os
from sklearn.model_selection import train_test_split


def parameter_search_GLOBAL(paddVal, numConvLayers, numFilters, kernelSize, numFcLayers,
	neurons, batch, epochs, dropRate, learnRate, classificType,aaSize,modelNum):
		for numConvL in numConvLayers:
			for numFcL in numFcLayers:
				for kernel in kernelSize:
					for learn in learnRate:
						# for classif in classificType:
						# para a camada 1 avalia 2 numfiltros diferentes, camada 2 avalia 4 combinações diferentes e para 3 avalia 8 combinações de filtros diferentes
						for i in range(0,2**numConvL):
							filters=random.sample(numFilters,k=numConvL)
							for j in range(0,2**numFcL):
								neur=random.sample(neurons,k=numFcL)
								for drop in dropRate:
									print('\nWith Dropout:')
									path="models/savedModels/model_"+str(modelNum)+".h5"
									modelName=str(modelNum)+'_GLOBAL_Conv:'+str(filters)+'-'+str(kernel)+"x"+str(kernel)+"_FC:"+str(neur)+"_Drop:"+str(drop)+"_Learn:"+str(learn)
									modelNum+=1
									print(modelName)
									model=CnnModel(paddVal,numConvL,filters,kernel,[],numFcL,neur,batch,epochs,drop,learn,'FC',path,aaSize, 'global', 'yes', 'fc')
									save_results([modelName]+model.results)
								# print('\nWithout Dropout:')
								# path="models/savedModels/model_"+str(modelNum)+".h5"
								# modelName=str(modelNum)+'_GLOBAL'+"_Conv:"+str(filters)+'-'+str(kernel)+"x"+str(kernel)+"_FC:"+str(neur)+"_Drop:NO"+"_Learn:"+str(learn)
								# modelNum+=1
								# print(modelName)
								# model=CnnModel(paddVal,numConvL,filters,kernel,[],numFcL,neur,batch,epochs,drop,learn,'FC',path,aaSize, 'global', 'no', 'fc')
								# save_results([modelName]+model.results)

def parameter_search_GAP(paddVal, numConvLayers, numFilters, kernelSize, poolSize, 
	batch, epochs, learnRate, classificType,aaSize, modelNum):
		for numConvL in numConvLayers:
			for kernel in kernelSize:
				for learn in learnRate:
					# for classif in classificType:
							# para a camada 1 avalia 2 numfiltros diferentes, camada 2 avalia 4 combinações diferentes e para 3 avalia 8 combinações de filtros diferentes
					for i in range(0,2**numConvL):
						filters=random.sample(numFilters,k=numConvL)
						for pool in poolSize:
							path="models/savedModels/model_"+str(modelNum)+".h5"
							modelName=str(modelNum)+'_GAP'+"_Conv:"+str(filters)+'-'+str(kernel)+"x"+str(kernel)+'_Pool:'+str(pool)+"_Learn:"+str(learn)
							modelNum+=1
							print(modelName)
							model=CnnModel(paddVal,numConvL,filters,kernel,pool,[],[],batch,epochs,[],learn,'FC',path,aaSize, 'local', 'no', 'gap')
							save_results([modelName]+model.results)

def parameter_search_LOCAL(paddVal,activFun, numConvLayers, numFilters, kernelSize, poolSize, numFcLayers,
	neurons, batch, epochs, dropRate, learnRate, classificType,aaSize, modelNum, archNum):
		for numConvL in numConvLayers:
			for numFcL in numFcLayers:
				for kernel in kernelSize:
					for learn in learnRate:
						# for classif in classificType:
						# para a camada 1 avalia 2 numfiltros diferentes, camada 2 avalia 4 combinações diferentes e para 3 avalia 8 combinações de filtros diferentes
						for i in range(0,2**numConvL):
							filters=random.sample(numFilters,k=numConvL)
							for j in range(0,2**numFcL):
								neur=random.sample(neurons,k=numFcL)
								for pool in poolSize:
									for drop in dropRate:
										for l in range(1,5,3):
											# To test different datasets
											# l=1 - dataset is the negatives
											# l=2 - dataset is the benchmark (suppAB)
											# l=3 - dataset is the benchmark (suppCD)
											# l=4 - dataset is the benchmark (suppE)
											print('\nDATA: '+str(l))
											print('\nOPT:1')

											# Optimizer 1 - adam
											# Relu activation Function
											path="models/savedModels/model_"+str(modelNum)+".h5"
											modelName=str(modelNum)+'_LOCAL'+"_Conv:"+str(filters)+'-'+str(kernelSize)+"x"+str(kernelSize)+'_Pool:'+str(poolSize)+"_FC:"+str(neur)+"_ActivFun:"+str(activFun[0])+"_Drop:"+str(dropRate)+"_Learn:"+str(learnRate)+"_Opt:"+str(1)+"_DataUsed:"+str(l)	
											print(modelName)
											model=CnnModel(paddVal[l-1],activFun[0],numConvL,filters,kernel,pool,numFcL,neur,batch,epochs,drop,learn,path,aaSize,archNum,1,1,l)
											save_results([modelName]+model.results)
											modelNum+=1
											# Swish activation Function
											path="models/savedModels/model_"+str(modelNum)+".h5"
											modelName=str(modelNum)+'_LOCAL'+"_Conv:"+str(filters)+'-'+str(kernelSize)+"x"+str(kernelSize)+'_Pool:'+str(poolSize)+"_FC:"+str(neur)+"_ActivFun:"+str(activFun[1])+"_Drop:"+str(dropRate)+"_Learn:"+str(learnRate)+"_Opt:"+str(1)+"_DataUsed:"+str(l)	
											print(modelName)
											model=CnnModel(paddVal[l-1],activFun[1],numConvL,filters,kernel,pool,numFcL,neur,batch,epochs,drop,learn,path,aaSize,archNum,1,1,l)
											save_results([modelName]+model.results)
											modelNum+=1

											# Optimizer 3 - rmsProp
											# Relu activation Function
											print('\nOPT:3')
											path="models/savedModels/model_"+str(modelNum)+".h5"
											modelName=str(modelNum)+'_LOCAL'+"_Conv:"+str(filters)+'-'+str(kernelSize)+"x"+str(kernelSize)+'_Pool:'+str(poolSize)+"_FC:"+str(neur)+"_ActivFun:"+str(activFun[0])+"_Drop:"+str(dropRate)+"_Learn:"+str(learnRate)+"_Opt:"+str(3)+"_DataUsed:"+str(l)
											print(modelName)
											model=CnnModel(paddVal[l-1],activFun[0],numConvL,filters,kernel,pool,numFcL,neur,batch,epochs,drop,learn,path,aaSize,archNum,1,3,l)
											save_results([modelName]+model.results)
											modelNum+=1
											# Swish activation Function
											path="models/savedModels/model_"+str(modelNum)+".h5"
											modelName=str(modelNum)+'_LOCAL'+"_Conv:"+str(filters)+'-'+str(kernelSize)+"x"+str(kernelSize)+'_Pool:'+str(poolSize)+"_FC:"+str(neur)+"_ActivFun:"+str(activFun[1])+"_Drop:"+str(dropRate)+"_Learn:"+str(learnRate)+"_Opt:"+str(3)+"_DataUsed:"+str(l)
											print(modelName)
											model=CnnModel(paddVal[l-1],activFun[1],numConvL,filters,kernel,pool,numFcL,neur,batch,epochs,drop,learn,path,aaSize,archNum,1,3,l)
											save_results([modelName]+model.results)
											modelNum+=1


									
def write_to_txt(fileName,seqA,seqB,label):
	os.makedirs(os.path.dirname(fileName), exist_ok=True)
	with open(fileName, 'w') as file:
		for a,b,c in zip(seqA,seqB,label):
			file.write(str(a)+'\t'+str(b)+'\t'+str(c)+'\n')

def limitate_length_new_files(fileName1, fileName2):
	# positive data
	with open(fileName1, 'r') as file:
		text1=file.read().split('>')
		proteins1=text1[1:]
	seqA1=[proteins1[i].split('\n')[1] for i in range(0,len(proteins1),2)]
	seqB1=[proteins1[i].split('\n')[1] for i in range(1,len(proteins1),2)]
	labels1=np.ones((1,len(seqA1)),dtype=int)
	
	# negative data
	with open(fileName2, 'r') as file:
		text2=file.read().split('>')
		proteins2=text2[1:]
	seqA2=[proteins2[i].split('\n')[1] for i in range(0,len(proteins2),2)]
	seqB2=[proteins2[i].split('\n')[1] for i in range(1,len(proteins2),2)]
	labels2=np.zeros((1,len(seqA2)),dtype=int)

	# Labels
	labels=np.concatenate((labels1,labels2),axis=1)

	# Padding
	lens=[len(p[i]) for i in range(len(seqA1+seqA2)) for p in (seqA1+seqA2,seqB1+seqB2)]
	maxLen=int(np.percentile(lens,90))
	# maxLen=int(max(lens))
	print("\nPadding value: ",maxLen)

	aminoacids=["C", "E", "L", "T", "I", "P", "A", "W", "R", "G", "V", "M", "H", "F", "Y", "Q", "N", "D", "S", "K", "U"]

	# Delete interactions with protein sequences that exceed maxLen and 
	# one of the amino of one of the sequences does not belong to the 21 amino considered
	ind=[i for i in range(len(seqA1+seqA2))]
	data=[i for i in ind if all(len(prot)<=maxLen for prot in ((seqA1+seqA2)[i],(seqB1+seqB2)[i]))]
	data=[i for i in data if all(amino in aminoacids for amino in (*(seqA1+seqA2)[i],*(seqB1+seqB2)[i]))]
	seqA=[(seqA1+seqA2)[i] for i in data]
	seqB=[(seqB1+seqB2)[i] for i in data]
	labels=np.array([labels[0][i] for i in data])

	# Split train and test
	ind=[i for i in range(len(seqA))]
	# labels=labels.reshape(labels.shape[1:])
	indTrain, indTest, yTrain, yTest = train_test_split(ind, labels, test_size=0.3, shuffle=True)

	seqTrainA=[seqA[i] for i in indTrain]
	seqTrainB=[seqB[i] for i in indTrain]
	seqTestA=[seqA[i] for i in indTest]
	seqTestB=[seqB[i] for i in indTest]
	# print(np.shape(seqTrainA))


	print("Train: "+ str(len(seqTrainA))+ " interactions")
	print("Test: "+ str(len(seqTestA))+ " interactions")

	write_to_txt("models/suppPaddedTrainCD.txt", seqTrainA,seqTrainB,yTrain)
	write_to_txt("models/suppPaddedTestCD.txt", seqTestA,seqTestB,yTest)
	return maxLen

def limitate_length_new_files2(fileName1):
	# positive data
	with open(fileName1, 'r') as file:
		text1=file.read().split('>')
		proteins1=text1[1:]
	seqA=[proteins1[i].split('\n')[1] for i in range(0,len(proteins1),2)]
	seqB=[proteins1[i].split('\n')[1] for i in range(1,len(proteins1),2)]
	labels1=np.ones((1,int(len(seqA)/2)),dtype=int)
	labels2=np.zeros((1,int(len(seqA)/2)),dtype=int)

	# Labels
	labels=np.concatenate((labels1,labels2),axis=1)
	
	# Padding
	lens=[len(p[i]) for i in range(len(seqA)) for p in (seqA,seqB)]
	maxLen=int(np.percentile(lens,90))
	# maxLen=int(max(lens))
	print("\nPadding value: ",maxLen)

	aminoacids=["C", "E", "L", "T", "I", "P", "A", "W", "R", "G", "V", "M", "H", "F", "Y", "Q", "N", "D", "S", "K", "U"]

	# Delete interactions with protein sequences that exceed maxLen and 
	# one of the amino of one of the sequences does not belong to the 21 amino considered
	ind=[i for i in range(len(seqA))]
	data=[i for i in ind if all(len(prot)<=maxLen for prot in ((seqA)[i],(seqB)[i]))]
	data=[i for i in data if all(amino in aminoacids for amino in (*(seqA)[i],*(seqB)[i]))]
	seqA=[(seqA)[i] for i in data]
	seqB=[(seqB)[i] for i in data]
	labels=np.array([labels[0][i] for i in data])

	# Split train and test
	ind=[i for i in range(len(seqA))]
	# labels=labels.reshape(labels.shape[1:])
	indTrain, indTest, yTrain, yTest = train_test_split(ind, labels, test_size=0.3, shuffle=True)
	print(np.shape(seqA))
	seqTrainA=[seqA[i] for i in indTrain]
	seqTrainB=[seqB[i] for i in indTrain]
	seqTestA=[seqA[i] for i in indTest]
	seqTestB=[seqB[i] for i in indTest]

	print("Train: "+ str(len(seqTrainA))+ " interactions")
	print("Test: "+ str(len(seqTestA))+" interactions")

	write_to_txt("models/suppPaddedTrainE.txt", seqTrainA,seqTrainB,yTrain)
	write_to_txt("models/suppPaddedTestE.txt", seqTestA,seqTestB,yTest)
	return maxLen

def load_data_DeepPPI(fileName1):
	with open(fileName1, 'r') as file:
		text=file.read().split('\n')
		pairs=text[1:len(text)-1]

	protA=[pairs[i].split(',')[0] for i in range(len(pairs))]
	protB=[pairs[i].split(',')[1] for i in range(len(pairs))]
	label=[pairs[i].split(',')[2] for i in range(len(pairs))]

	uniqueProts=list(set(protA+protB))
	print("Unique proteins: "+str(len(uniqueProts)))
	
	# os.makedirs(os.path.dirname("files/mappingSequencesDeepPPI.txt"), exist_ok=True)
	# with open("files/mappingSequencesDeepPPI.txt", 'w') as file:
	# 	for a in uniqueProts:
	# 		file.write(str(a)+'\n')
	
	file=open('files/'+"sequencesDeepPPIProteins.fasta", 'r').read().split('>')
	data=file[:]
	info=[prot.split('\n',1)[0] for prot in data if len(prot.split('\n',1))>1]
	seqs=[prot.split('\n',1)[1] for prot in data if len(prot.split('\n',1))>1]
	# Dic structure UniProtID:sequence
	dicSeqs={inf[inf.find('|')+1:inf.find('|',inf.find('|')+1)]: seq.replace('\n','') for inf, seq in zip(info,seqs)}
	# Padding
	lens=[len(dicSeqs[p]) for p in (protA+protB)]
	maxLen=int(np.percentile(lens,90))
	# maxLen=int(max(lens))
	print("\nPadding value: ",maxLen)
	print("\nPre-padding number interactions: "+str(len(protA)))
	seqA=[dicSeqs[protA[i]] for i in range(len(protA)) if len(dicSeqs[protA[i]])<=maxLen and len(dicSeqs[protB[i]])<=maxLen]
	seqB=[dicSeqs[protB[i]] for i in range(len(protA)) if len(dicSeqs[protA[i]])<=maxLen and len(dicSeqs[protB[i]])<=maxLen]
	labels=[label[i] for i in range(len(protA)) if len(dicSeqs[protA[i]])<=maxLen and len(dicSeqs[protB[i]])<=maxLen]

	print("\nPos-padding number interactions: "+str(len(seqA)))

	indexPositivePairs=[i for i in range(len(labels)) if labels[i]=='1']
	indexNegativePairs=[i for i in range(len(labels)) if labels[i]=='0']

	indexRandom=[]
	while (len(indexRandom)<len(indexPositivePairs)):
		i=random.randint(indexNegativePairs[0], indexNegativePairs[len(indexNegativePairs)-1])
		if i not in indexRandom: indexRandom.append(i)

	seqA=[seqA[i] for i in (indexPositivePairs+indexRandom)]
	seqB=[seqB[i] for i in (indexPositivePairs+indexRandom)]
	labels=[labels[i] for i in (indexPositivePairs+indexRandom)]
	
	# Split train and test
	ind=[i for i in range(len(seqA))]
	# labels=labels.reshape(labels.shape[1:])
	indTrain, indTest, yTrain, yTest = train_test_split(ind, labels, test_size=0.3, shuffle=True)
	print(np.shape(seqA))
	seqTrainA=[seqA[i] for i in indTrain]
	seqTrainB=[seqB[i] for i in indTrain]
	seqTestA=[seqA[i] for i in indTest]
	seqTestB=[seqB[i] for i in indTest]

	print("Train: "+ str(len(seqTrainA))+ " interactions")
	print("Test: "+ str(len(seqTestA))+" interactions")

	write_to_txt("models/deepPPIPaddedTrain.txt", seqTrainA,seqTrainB,yTrain)
	write_to_txt("models/deepPPIPaddedTest.txt", seqTestA,seqTestB,yTest)
	return maxLen

def pre_process_ProfPPI_data(dirName):
	# https://rostlab.org/owiki/index.php/More_challenges_for_machine_learning_protein_protein_interactions
	# https://www.ncbi.nlm.nih.gov/pubmed/25657331
	# the C1 difficulty was chosen, and the non-redundant sequences
	# Get the training data, 9 out of the 10 partitions
	allPairs=[]
	label=[]
	for i in range(1,10):
		fileName1=dirName+'/'+str(i)+'.train.neg'
		with open(fileName1, 'r') as file:
			text=file.read().split('\n')
			pairs1=text[:len(text)-1]
			# pairs1=text[:3]
			allPairs[len(allPairs):len(allPairs)]=pairs1
			label[len(label):len(label)]=[0]*len(pairs1)

		fileName2=dirName+'/'+str(i)+'.train.pos'
		with open(fileName2, 'r') as file:
			text=file.read().split('\n')
			pairs2=text[:len(text)-1]
			# pairs2=text[:3]
			allPairs[len(allPairs):len(allPairs)]=pairs2
			label[len(label):len(label)]=[1]*len(pairs2)

	testPairs=[]
	testLabels=[]
	fileName1=dirName+'/0.train.neg'
	with open(fileName1, 'r') as file:
		text=file.read().split('\n')
		pairs1=text[:3]
		testPairs[len(testPairs):len(testPairs)]=pairs1
		testLabels[len(testLabels):len(testLabels)]=[0]*len(pairs1)

	fileName2=dirName+'/0.train.pos'
	with open(fileName2, 'r') as file:
		text=file.read().split('\n')
		pairs1=text[:len(text)-1]
		testPairs[len(testPairs):len(testPairs)]=pairs1
		testLabels[len(testLabels):len(testLabels)]=[0]*len(pairs1)

	protA=[allPairs[i].split(' ')[0] for i in range(len(allPairs))]
	protB=[allPairs[i].split(' ')[1] for i in range(len(allPairs))]
	testProtA=[testPairs[i].split(' ')[0] for i in range(len(testPairs))]
	testProtB=[testPairs[i].split(' ')[1] for i in range(len(testPairs))]
	
	# # verify if there are duplicates
	# c=[]
	# for i in range(len(protA)):
	# 	a=0
	# 	for j in range(len(protA)):
	# 		# condition for duplicates
	# 		if protA[i]==protA[j] and protB[i]==protB[j]:
	# 			a=a+1
	# 			if a>1:
	# 				del protA[i]
	# 				del protB[i]
	# 		# condition for redundants
	# 		elif protA[i]==protB[j] and protB[i]==protA[j]:
	# 			a=a+1
	# 			if a>1:
	# 				del protA[i]
	# 				del protB[i]
	# print(protA)
	# print("\nPre duplication removal number interactions: "+str(len(protA)))
	# # removing duplicates
	# for ind in sorted(c, reverse = True):
	# 	del protA[ind]
	# 	del protB[ind]
	# print("\nPos duplciation removal number interactions: "+str(len(protA)))
	# print(protA)

	file=open("data/split_0/fasta.txt", 'r').read().split('>')
	data=file[1:]
	
	names=[prot.split('\n')[0] for prot in data]
	seqs=[prot.split('\n')[1] for prot in data]
	# Dic structure UniProtID:sequence
	dicSeqs={name: seq for name, seq in zip(names,seqs)}

	lens=[len(dicSeqs[p]) for p in (protA+protB+testProtA+testProtB)]
	maxLen=int(np.percentile(lens,90))
	# maxLen=int(max(lens))
	print("\nPadding value: ",maxLen)
	print("\nPre-padding number interactions: "+str(len(protA)))
	print("\nPre-padding number interactions test: "+str(len(testProtA)))
	seqTrainA=[dicSeqs[protA[i]] for i in range(len(protA)) if len(dicSeqs[protA[i]])<=maxLen and len(dicSeqs[protB[i]])<=maxLen]
	seqTrainB=[dicSeqs[protB[i]] for i in range(len(protA)) if len(dicSeqs[protA[i]])<=maxLen and len(dicSeqs[protB[i]])<=maxLen]
	yTrain=[label[i] for i in range(len(protA)) if len(dicSeqs[protA[i]])<=maxLen and len(dicSeqs[protB[i]])<=maxLen]

	seqTestA=[dicSeqs[testProtA[i]] for i in range(len(testProtA)) if len(dicSeqs[testProtA[i]])<=maxLen and len(dicSeqs[testProtB[i]])<=maxLen]
	seqTestB=[dicSeqs[testProtB[i]] for i in range(len(testProtB)) if len(dicSeqs[testProtA[i]])<=maxLen and len(dicSeqs[testProtB[i]])<=maxLen]
	yTest=[testLabels[i] for i in range(len(testProtA)) if len(dicSeqs[testProtA[i]])<=maxLen and len(dicSeqs[testProtB[i]])<=maxLen]

	print("\nPos-padding number interactions: "+str(len(seqTrainA)))
	print("\nPos-padding number interactions test: "+str(len(seqTestA)))

	write_to_txt("models/deepProfPPIPaddedTrain.txt", seqTrainA,seqTrainB,yTrain)
	write_to_txt("models/deepProfPPIPaddedTest.txt", seqTestA,seqTestB,yTest)
	return maxLen
	



	

def save_results(results):
		# Save results to csv
		with open('models/model_results.txt', mode='a') as file:
			# modelName, Loss, Acc, Sens, Spec, trainAcc, trainSens, valAcc, valSens 
			file.write(results[0]+'\tLoss:'+results[1]+'\tAcc:'+results[2]+'\tSens:'+results[3]+'\tSpec:'+results[4]+'\tPrecision:'+results[5]+'\tf1Score:'+results[6]+'\ttrainAcc:'+results[6]+'\tvalAcc:'+results[8]+'\n')

if __name__ == "__main__":


	# -------------------------------------------------------------------------------------------------
	# Dataset statistics
	#
	# For the benchmark data (A-B), after some pre-processing
	# 59545 interactions: 41681 train (SuppPaddedTrain.txt) and 17864 test (SuppPaddedTrain.txt)
	# paddVal=1141  
	# 
	# For the benchmark data (C-D), after some pre-processing
	# 6657 interactions: 4659 train (SuppPaddedTrain.txt) and 1998 test (SuppPaddedTrain.txt)
	# paddVal=1214
	#
	# For the benchmark data (E), after some pre-processing
	# 1530 interactions: 1071 train (SuppPaddedTrain.txt) and 459 test (SuppPaddedTrain.txt)
	# paddVal=1039
	# 
	# For the benchmark data of DeepPPI, after some pre-processing 
	# 27858 interactions: 19500 train (deepPPIPaddedTrain.txt) and 8358 test (deepPPIPaddedTest.txt)
	# paddVal=1030
	# 
	# For the benchmark data of ProfPPI, after some pre-processing 
	# 61000 interactions: 60873 train (deepPPIPaddedTrain.txt) and 634 test (deepPPIPaddedTest.txt)
	# paddVal=1202
	# 
	# For the whole dataset, not filtered, with my UniprotInteractions 
	# 63221 positive interactions: 36092 train (positiveTrainNoFilter.txt) and 15468 test (positiveTestNoFilter.txt)
	# paddVal=1215 
	#
	# For the whole physical dataset creating the negatives with only the positive physical data, not filtered, with my UniprotInteractions 
	# 36244 positive interactions: 25370 train (positiveTrainNoFilterPhysical.txt) and 10873 test (positiveTestNoFilterPhysical.txt)
	# paddVal=1231
	# 
	# For the whole physical dataset creating the negatives with the whole positive data, not filtered, with my UniprotInteractions 
	# 36244 positive interactions: 25370 train (positiveTrainNoFilterPhysicalAllPos.txt) and 10873 test (positiveTestNoFilterPhysicalAllPos.txt)
	# paddVal=1231
	# 
	# For the filtered data by two GO () with my Uniprot interactions 
	# 23369 positive interactions: 16358 train (positiveTrain.txt) and 7010 test (positiveTest.txt)
	# paddVal=1367
	#
	# For the filtered data by two GO () with teacher's interactions
	# positive interactions: 24554 train (positiveTrainTeacher.txt) and 10523 test (positiveTestTeacher.txt)
	# paddVal=1267


	# -------------------------------------------------------------------------------------------------
	# Load Data and split in train and test

	# If data files need to be created uncomment the 2 following lines
	# Filtering by GO
	# a=DataCreator(['chromatinbinding0003682.fasta','organiccycliccompoundbinding0097159.fasta'],1,2)

	# Whole data
	# a=DataCreator(['homoProteins.fasta'],2,2)
	# b=DataSplitter(a)

	# Benchmark data set
	# needed some pre-processing in order to eliminate certain sequences that were outliers, 
	# due to their huge number of sequences (ex: there were sequences with 34000 amino acids)
	# paddVal1=limitate_length_new_files('models/Supp-C.txt','models/Supp-D.txt')
	# paddVal2=limitate_length_new_files2('models/Supp-E.txt')
	# load_data_DeepPPI("files/Supplementary S1.csv")
	# pre_process_ProfPPI_data("data/split_0")


	# -------------------------------------------------------------------------------------------------
	# Parameter Search
	paddVal=[1215,1141,1214, 1039,1231,1231,1030,1202]
	numFilters=[64,128,160,256]
	# numFilters=[64,128,256,512,1024]
	kernelSize=[2]
	# kernelSize=[2,3,5]
	actFun=['relu','swish']
	#dropRate=[0.1,0.3]
	dropRate=[0.1]
	# batch=20
	# fcNeurons=[128,256,512,1024]
	fcNeurons=[128,256,512]
	# numConvLayers=[2, 3, 4, 5]
	numConvLayers=[3]
	# numFcLayers=[1, 2, 3]
	numFcLayers=[3]
	#learnRate=[0.001,0.01]
	learnRate=[0.001]
	classificType=['FC']
	padding='same'
	stride=1
	epochs=100
	batch=128
	#poolSize=[2,3]
	poolSize=[2]
	dictionarySize=20
	archNum=4

	# RESNET ARCHITECTURE
	# modelNum=9
	# path="models/savedModels/model_"+str(modelNum)+".h5"
	# modelName=str(modelNum)+'_RESNET'+"_Opt:"+str(1)+"__DataUsed:"+str(4)
	# modelNum+=1
	# model=CnnModel(paddVal[0],len(numFilters),numFilters,kernelSize[0],poolSize[0],len(fcNeurons),fcNeurons,batch,epochs,dropRate[0],learnRate[0],'FC',path,dictionarySize, 'local', 'yes', 'fc',4,1,1,4)
	# save_results([modelName]+model.results)

	# path="models/savedModels/model_"+str(modelNum)+".h5"
	# modelName=str(modelNum)+'_RESNET'+"_Opt:"+str(1)+"__DataUsed:"+str(1)
	# model=CnnModel(paddVal[0],len(numFilters),numFilters,kernelSize[0],poolSize[0],len(fcNeurons),fcNeurons,batch,epochs,dropRate[0],learnRate[0],'FC',path,dictionarySize, 'local', 'yes', 'fc',4,1,1,1)
	# save_results([modelName]+model.results)

	# According to the architecture that u want to hyperparameter tuning, unccomment the one you prefer
	# 1) Global:
	#		[Convolutional]^n -> GlobalMaxPooling ->[FC]^m -> OutputLayer
	# 2) Gap:
	#		[Convolutional -> MaxPooling]^n -> GlobalAveragePooling -> OutputLayer
	# 3) Local (the standard approach):
	#		[Convolutional -> MaxPooling]^n -> [FC]^m -> OutputLayer

	# parameter_search_GLOBAL(paddVal, numConvLayers, numFilters, kernelSize, numFcLayers,
	# fcNeurons, batch, epochs, dropRate, learnRate, classificType, dictionarySize, 1)
	# parameter_search_GAP(paddVal, numConvLayers, numFilters, kernelSize, poolSize, 
	# batch, epochs, learnRate, classificType, dictionarySize,2)
	# parameter_search_LOCAL(paddVal,actFun, numConvLayers, numFilters, kernelSize, poolSize, numFcLayers,
	# fcNeurons, batch, epochs, dropRate, learnRate, classificType, dictionarySize,11, archNum)


	# -------------------------------------------------------------------------------------------------
	# To run only one model, with the parameters chosen by the user

	# modelName=str(modelNum)+'_LOCAL'+"_Conv:"+str(filters)+'-'+str(kernelSize)+"x"+str(kernelSize)+'_Pool:'+str(poolSize)+"_FC:"+str(neur)+"_Drop:"+str(dropRate)+"_Learn:"+str(learnRate)+"_Opt:"+str(i)

	# model=CnnModel(paddVal[0],actFun[0],len(filters),filters,kernelSize[0],poolSize[0],len(neur),neur,batch,epochs,dropRate[0],learnRate[0],'FC',path,dictionarySize, 'local', 'yes', 'fc',1,1,3,4)
	# save_results([modelName]+model.results)
	dataToLoad=7
	filters=[128,256,512]
	neur=[256,512,1024]
	modelNum=21
	opt=1
	archNum=3
	concatOrMulti='concatenate'

	# path="models/savedModels/model_"+str(modelNum)+".h5"
	# modelName=str(modelNum)+'_LOCAL'+"_Conv:"+str(filters)+'-'+str(kernelSize[0])+"x"+str(kernelSize[0])+'_Pool:'+str(poolSize[0])+"_FC:"+str(neur)+"_ActivFun:"+str(actFun[0])+"_Drop:"+str(dropRate[0])+"_Learn:"+str(learnRate[0])+"_Opt:"+str(opt)+"_DataUsed:"+str(dataToLoad)+'_'+str(concatOrMulti)
	# print(modelName)
	# model=CnnModel(paddVal[dataToLoad-1],actFun[0],len(filters),filters,kernelSize[0],poolSize[0],len(neur),neur,batch,epochs,dropRate[0],learnRate[0],path,dictionarySize,archNum,1,opt,dataToLoad,concatOrMulti)
	# save_results([modelName]+model.results)

	modelNum=modelNum+1
	path="models/savedModels/model_"+str(modelNum)+".h5"
	modelName=str(modelNum)+'_LOCAL'+"_Conv:"+str(filters)+'-'+str(kernelSize[0])+"x"+str(kernelSize[0])+'_Pool:'+str(poolSize[0])+"_FC:"+str(neur)+"_ActivFun:"+str(actFun[1])+"_Drop:"+str(dropRate)+"_Learn:"+str(learnRate)+"_Opt:"+str(3)+"_DataUsed:"+str(dataToLoad)+'_'+str(concatOrMulti)
	print(modelName)
	model=CnnModel(paddVal[dataToLoad-1],actFun[0],len(filters),filters,kernelSize[0],poolSize[0],len(neur),neur,batch,epochs,dropRate[0],learnRate[0],path,dictionarySize,archNum,1,3,dataToLoad,concatOrMulti)
	save_results([modelName]+model.results)


	modelNum=modelNum+1
	path="models/savedModels/model_"+str(modelNum)+".h5"
	modelName=str(modelNum)+'_ConvLstm'+"_Conv:"+str(filters)+'-'+str(kernelSize[0])+"x"+str(kernelSize[0])+'_Pool:'+str(poolSize[0])+'_LSTM:80'+"_ActivFun:"+str(actFun[0])+"_Learn:"+str(learnRate[0])+"_Opt:"+str(opt)+"_DataUsed:"+str(dataToLoad)+'_'+str(concatOrMulti)
	print(modelName)
	model=ConvLstm(paddVal[dataToLoad-1],filters,kernelSize[0],actFun[0],poolSize[0],batch,epochs,dropRate[0],learnRate[0],path,dictionarySize,opt,dataToLoad,concatOrMulti)
	save_results([modelName]+model.results)

	modelNum=modelNum+1
	path="models/savedModels/model_"+str(modelNum)+".h5"
	modelName=str(modelNum)+'_FullyCnnPooling'+"_Conv:"+str(filters)+'-'+str(kernelSize[0])+"x"+str(kernelSize[0])+'_Pool:'+str(poolSize[0])+"_ActivFun:"+str(actFun[0])+"_Learn:"+str(learnRate[0])+"_Opt:"+str(opt)+"_DataUsed:"+str(dataToLoad)+'_'+str(concatOrMulti)
	print(modelName)
	model=FullyCnnModel(paddVal[dataToLoad-1],filters,kernelSize[0],actFun[0],poolSize[0],batch,epochs,dropRate[0],learnRate[0],path,dictionarySize,opt,dataToLoad,'pool',concatOrMulti)
	save_results([modelName]+model.results)

	modelNum=modelNum+1
	path="models/savedModels/model_"+str(modelNum)+".h5"
	modelName=str(modelNum)+'_FullyCnn'+"_Conv:"+str(filters)+'-'+str(kernelSize[0])+"x"+str(kernelSize[0])+'_Pool:'+str(poolSize[0])+"_ActivFun:"+str(actFun[0])+"_Learn:"+str(learnRate[0])+"_Opt:"+str(opt)+"_DataUsed:"+str(dataToLoad)+'_'+str(concatOrMulti)
	print(modelName)
	model=FullyCnnModel(paddVal[dataToLoad-1],filters,kernelSize[0],actFun[0],poolSize[0],batch,epochs,dropRate[0],learnRate[0],path,dictionarySize,opt,dataToLoad,'no',concatOrMulti)
	save_results([modelName]+model.results)

	# modelNum=modelNum+1
	# path="models/savedModels/model_"+str(modelNum)+".h5"
	# modelName=str(modelNum)+'_Ensemble'+"_FC:"+str(neur)+"_ActivFun:"+str(actFun[0])+"_Drop:"+str(dropRate[0])+"_Learn:"+str(learnRate[0])+"_Opt:"+str(opt)+"_DataUsed:"+str(dataToLoad)
	# print(modelName)
	# model=EnsembleModel(paddVal[dataToLoad-1],actFun[0],neur,batch,epochs,dropRate[0],learnRate[0],path,dictionarySize,opt,dataToLoad)
	# save_results([modelName]+model.results)

	# -------------------------------------------------------------------------------------------------
	# USING BECHMARK MODEL ON MY DATA
	
	# dataToLoad=6
	# autoEncoderNeurs=[400,420]
	# opt=2
	# learnRate=[1]
	# modelNum=modelNum+1
	# path1="models/savedModels/autoencoder_"+str(modelNum)+".h5"
	# path2="models/savedModels/model_"+str(modelNum)+".h5"
	# modelName=str(modelNum)+'_Autoencoder'+"_FC:"+str(autoEncoderNeurs)+"_ActivFun:"+str(actFun[0])+"_Learn:"+str(learnRate[0])+"_Opt:"+str(opt)+"_DataUsed:"+str(dataToLoad)
	# print(modelName)
	# model=AutoEncoderModel(paddVal[dataToLoad-1],actFun[0],autoEncoderNeurs,batch,epochs,learnRate[0],path1,path2,dictionarySize,opt,dataToLoad)
	# save_results([modelName]+model.results)

	# autoEncoderFilters=[400]
	# modelNum=modelNum+1
	# path1="models/savedModels/autoencoder_"+str(modelNum)+".h5"
	# path2="models/savedModels/model_"+str(modelNum)+".h5"
	# modelName=str(modelNum)+'_Autoencoder'+"_FC:"+str(autoEncoderNeurs)+"_ActivFun:"+str(actFun[0])+"_Learn:"+str(learnRate[0])+"_Opt:"+str(opt)+"_DataUsed:"+str(dataToLoad)
	# print(modelName)
	# model=AutoEncoderModel(paddVal[dataToLoad-1],actFun[0],autoEncoderNeurs,batch,epochs,learnRate[0],path1,path2,dictionarySize,opt,dataToLoad)
	# save_results([modelName]+model.results)