import random
import numpy as np
import math
import os
from pydpi import pypro
import shutil

# needs to be run in python 2.x


def load_data(fileName1, fileName2):
	# loads positive data
	with open(fileName1, 'r') as file:
		data1=file.read().split('\n')
		data1=data1[:len(data1)-1]
		# data1=data1[:5]
	# loads negative data
	with open(fileName2, 'r') as file:
		data2=file.read().split('\n')
		data2=data2[:len(data2)-1]
		# data2=data2[:5]
	# The files loaded have 3 collumns separated by a tab, has the following example: protA protB labelOfInteraction

	labels1=[i.split('\t')[4] for i in data1]
	labels2=[i.split('\t')[4] for i in data2]
	protA1=[i.split('\t')[0] for i in data1]
	protA2=[i.split('\t')[0] for i in data2]
	protB1=[i.split('\t')[1] for i in data1]
	protB2=[i.split('\t')[1] for i in data2]
	seqA1=[i.split('\t')[2] for i in data1]
	seqA2=[i.split('\t')[2] for i in data2]
	seqB1=[i.split('\t')[3] for i in data1]
	seqB2=[i.split('\t')[3] for i in data2]

	# Shuffle the input. Getting a random sequence of the indexes of the list
	ind=[i for i in range(0,len(labels1+labels2))]
	random.shuffle(ind)
	# Suffled labels
	labels=[int(i) for i in (labels1+labels2)]
	labels=[labels[i] for i in ind]
	seqA=[(seqA1+seqA2)[i] for i in ind]
	seqB=[(seqB1+seqB2)[i] for i in ind]
	protA=[(protA1+protA2)[i] for i in ind]
	protB=[(protB1+protB2)[i] for i in ind]
	return protA,protB,seqA,seqB,labels

def get_benchmark_data(fileName):
	# Benchmark data needed some pre-processing in order to eliminate certain sequences that were outliers, 
	# due to their huge number of sequences (ex: there were sequences with 34000 amino acids)
	with open(fileName, 'r') as file:
		data1=file.read().split('\n')
		data1=data1[:len(data1)-1]
		# data1=data1[:3]			
	
	# The files loaded have 3 collumns separated by a tab, has the following example: protA protB labelOfInteraction
	labels1=[int(i.split('\t')[2]) for i in data1]
	seqA=[i.split('\t')[0] for i in data1]
	seqB=[i.split('\t')[1] for i in data1]

	
	ind=[i for i in range(0,len(labels1))]
	random.shuffle(ind)
	# Suffled labels
	labels=[int(i) for i in (labels1)]
	labels=[labels[i] for i in ind]
	A=[seqA[i] for i in ind]
	B=[seqB[i] for i in ind]
	return A,B,labels

def ac_encode(A,B):
	# Taken from: https://github.com/smalltalkman/hppi-tensorflow/tree/master/coding
	# https://www.sciencedirect.com/science/article/pii/S0025556418307168
	# PCPNS: Physicochemical property names
	PCPNS = ['H1', 'H2', 'NCI', 'P1', 'P2', 'SASA', 'V']

	# AAPCPVS: Physicochemical property values of amino acid
	AAPCPVS = {
    'A': { 'H1': 0.62, 'H2':-0.5, 'NCI': 0.007187, 'P1': 8.1, 'P2':0.046, 'SASA':1.181, 'V': 27.5 },
    'C': { 'H1': 0.29, 'H2':-1.0, 'NCI':-0.036610, 'P1': 5.5, 'P2':0.128, 'SASA':1.461, 'V': 44.6 },
    'D': { 'H1':-0.90, 'H2': 3.0, 'NCI':-0.023820, 'P1':13.0, 'P2':0.105, 'SASA':1.587, 'V': 40.0 },
    'E': { 'H1': 0.74, 'H2': 3.0, 'NCI': 0.006802, 'P1':12.3, 'P2':0.151, 'SASA':1.862, 'V': 62.0 },
    'F': { 'H1': 1.19, 'H2':-2.5, 'NCI': 0.037552, 'P1': 5.2, 'P2':0.290, 'SASA':2.228, 'V':115.5 },
    'G': { 'H1': 0.48, 'H2': 0.0, 'NCI': 0.179052, 'P1': 9.0, 'P2':0.000, 'SASA':0.881, 'V':  0.0 },
    'H': { 'H1':-0.40, 'H2':-0.5, 'NCI':-0.010690, 'P1':10.4, 'P2':0.230, 'SASA':2.025, 'V': 79.0 },
    'I': { 'H1': 1.38, 'H2':-1.8, 'NCI': 0.021631, 'P1': 5.2, 'P2':0.186, 'SASA':1.810, 'V': 93.5 },
    'K': { 'H1':-1.50, 'H2': 3.0, 'NCI': 0.017708, 'P1':11.3, 'P2':0.219, 'SASA':2.258, 'V':100.0 },
    'L': { 'H1': 1.06, 'H2':-1.8, 'NCI': 0.051672, 'P1': 4.9, 'P2':0.186, 'SASA':1.931, 'V': 93.5 },
    'M': { 'H1': 0.64, 'H2':-1.3, 'NCI': 0.002683, 'P1': 5.7, 'P2':0.221, 'SASA':2.034, 'V': 94.1 },
    'N': { 'H1':-0.78, 'H2': 2.0, 'NCI': 0.005392, 'P1':11.6, 'P2':0.134, 'SASA':1.655, 'V': 58.7 },
    'P': { 'H1': 0.12, 'H2': 0.0, 'NCI': 0.239531, 'P1': 8.0, 'P2':0.131, 'SASA':1.468, 'V': 41.9 },
    'Q': { 'H1':-0.85, 'H2': 0.2, 'NCI': 0.049211, 'P1':10.5, 'P2':0.180, 'SASA':1.932, 'V': 80.7 },
    'R': { 'H1':-2.53, 'H2': 3.0, 'NCI': 0.043587, 'P1':10.5, 'P2':0.291, 'SASA':2.560, 'V':105.0 },
    'S': { 'H1':-0.18, 'H2': 0.3, 'NCI': 0.004627, 'P1': 9.2, 'P2':0.062, 'SASA':1.298, 'V': 29.3 },
    'T': { 'H1':-0.05, 'H2':-0.4, 'NCI': 0.003352, 'P1': 8.6, 'P2':0.108, 'SASA':1.525, 'V': 51.3 },
    'V': { 'H1': 1.08, 'H2':-1.5, 'NCI': 0.057004, 'P1': 5.9, 'P2':0.140, 'SASA':1.645, 'V': 71.5 },
    'W': { 'H1': 0.81, 'H2':-3.4, 'NCI': 0.037977, 'P1': 5.4, 'P2':0.409, 'SASA':2.663, 'V':145.5 },
    'Y': { 'H1': 0.26, 'H2':-2.3, 'NCI': 117.3000, 'P1': 6.2, 'P2':0.298, 'SASA':2.368, 'V':  0.023599 },
	}

	def avg_sd(NUMBERS):
		AVG = sum(NUMBERS)/len(NUMBERS)
		TEM = [pow(NUMBER-AVG, 2) for NUMBER in NUMBERS]
		DEV = sum(TEM)/len(TEM)
		SD = math.sqrt(DEV)
		return (AVG, SD)

	# PCPVS: Physicochemical property values
	PCPVS = {'H1':[], 'H2':[], 'NCI':[], 'P1':[], 'P2':[], 'SASA':[], 'V':[]}
	for AA, PCPS in AAPCPVS.items():
		for PCPN in PCPNS:
			PCPVS[PCPN].append(PCPS[PCPN])

	# PCPASDS: Physicochemical property avg and sds
	PCPASDS = {}
	for PCP, VS in PCPVS.items():
		PCPASDS[PCP] = avg_sd(VS)

	# NORMALIZED_AAPCPVS
	NORMALIZED_AAPCPVS = {}
	for AA, PCPS in AAPCPVS.items():
		NORMALIZED_PCPVS = {}
		for PCP, V in PCPS.items():
			NORMALIZED_PCPVS[PCP] = (V-PCPASDS[PCP][0])/PCPASDS[PCP][1]
		NORMALIZED_AAPCPVS[AA] = NORMALIZED_PCPVS

	def pcp_value_of(AA, PCP):
		"""Get physicochemical properties value of amino acid."""
		return NORMALIZED_AAPCPVS[AA][PCP];

	def pcp_sequence_of(PS, PCP):
		"""Make physicochemical properties sequence of protein sequence.
		"""
		PCPS = []
		for I, CH in enumerate(PS):
			PCPS.append(pcp_value_of(CH, PCP))
		# Centralization
		AVG = sum(PCPS)/len(PCPS)
		for I, PCP in enumerate(PCPS):
			PCPS[I] = PCP - AVG
		return PCPS

	def ac_values_of(PS, PCP, LAG):
		"""Get ac values of protein sequence."""
		AVS = []
		PCPS = pcp_sequence_of(PS, PCP)
		for LG in range(1, LAG+1):
			SUM = 0
			for I in range(len(PCPS)-LG):
				SUM = SUM + PCPS[I]*PCPS[I+LG]
			SUM = SUM / (len(PCPS)-LG)
			AVS.append(SUM)
		return AVS

	def all_ac_values_of(PS, LAG):
		"""Get all ac values of protein sequence."""
		AAVS = []
		for PCP in PCPS:
			AVS = ac_values_of(PS, PCP, LAG)
			AAVS = AAVS + AVS
		return AAVS

	def ac_code_of(PS):
		"""Get ac code of protein sequence."""
		AC_Code = all_ac_values_of(PS, 30)
		return AC_Code

	# print(len(ac_code_of('MKFVYKEEHPFEKRRSEGEKIRKKYPDRVPVIVEKAPKARIGDLDKKKYLVPSDLTVGQFYFLIRKRIHLRAEDALFFFVNNVIPPTSATMGQLYQEHHEEDFFLYIAYSDESVYGL')))

	# AC Values
	seqA=np.zeros((len(A),30*7))
	seqB=np.zeros((len(B),30*7))
	for i in range(len(A)):
		acA=ac_code_of(A[i])
		acB=ac_code_of(B[i])
		seqA[i,:]=acA
		seqB[i,:]=acB

	return seqA.astype('float32'),seqB.astype('float32')

def other_descriptors_using_pydpi(A,B):
	# https://pydoc.net/pydpi/1.0/pydpi.example/

	# for the conjoint triads
	seqA1=np.zeros((len(A),7*7*7))
	seqB1=np.zeros((len(B),7*7*7))
	seqA2=np.zeros((len(A),147))
	seqB2=np.zeros((len(B),147))
	proteinobject=pypro.PyPro()
	for i in range(len(A)):
		proteinobject.ReadProteinSequence(A[i])
		conjointDic=proteinobject.GetTriad()
		seqA1[i,:]=conjointDic.values()

		ctdDic=proteinobject.GetCTD()
		seqA2[i,:]=ctdDic.values()


		proteinobject.ReadProteinSequence(B[i])
		conjointDic=proteinobject.GetTriad()
		seqB1[i,:]=conjointDic.values()
		
		ctdDic=proteinobject.GetCTD()
		seqB2[i,:]=ctdDic.values()

	return seqA1, seqB1, seqA2, seqB2

def write_to_txt(fileName,protA,protB,valA,valB, label):
	dir = fileName
	# if os.path.exists(dir):
	# 	shutil.rmtree(dir,ignore_errors = True)
	# os.makedirs(dir)

	with open(fileName, 'w') as file:
		for a,b,c,d,e in zip(protA,protB,valA,valB,label):
			file.write(str(a)+'\t'+str(b)+'\t')
			for ci in c:
				file.write(str(ci)+',')
			file.write('\t')
			for di in d:
				file.write(str(di)+',')
			file.write('\t'+str(e)+'\n')

def write_to_txt_benchmark(fileName,valA,valB, label):

	with open(fileName, 'w') as file:
		for c,d,e in zip(valA,valB,label):
			for ci in c:
				file.write(str(ci)+',')
			file.write('\t')
			for di in d:
				file.write(str(di)+',')
			file.write('\t'+str(e)+'\n')

if __name__ == "__main__":

	dataToLoad=7

	if dataToLoad==1:
		protTrainA,protTrainB,trainA,trainB,labelTrain=load_data('data/positiveTrainNoFilter.txt','data/negativeTrainNoFilter.txt')
		protTestA,protTestB,testA,testB,labelTest=load_data('data/positiveTestNoFilter.txt','data/negativeTestNoFilter.txt')
	elif dataToLoad==2:
		protTrainA,protTrainB,trainA,trainB,labelTrain=get_benchmark_data('models/SuppPaddedTrain.txt')
		protTestA,protTestB,testA,testB,labelTest=get_benchmark_data('models/SuppPaddedTest.txt')
	elif dataToLoad==5:
		protTrainA,protTrainB,trainA,trainB,labelTrain=load_data('data/positiveTrainNoFilterPhysical.txt','data/negativeTrainNoFilterPhysical.txt')
		protTestA,protTestB,testA,testB,labelTest=load_data('data/positiveTestNoFilterPhysical.txt','data/negativeTestNoFilterPhysical.txt')
	elif dataToLoad==6:
		protTrainA,protTrainB,trainA,trainB,labelTrain=load_data('data/positiveTrainNoFilterPhysicalAllPos.txt','data/negativeTrainNoFilterPhysicalAllPos.txt')
		protTestA,protTestB,testA,testB,labelTest=load_data('data/positiveTestNoFilterPhysicalAllPos.txt','data/negativeTestNoFilterPhysicalAllPos.txt')
	elif dataToLoad==7:
		trainA,trainB,labelTrain=get_benchmark_data('models/deepPPIPaddedTrain.txt')
		testA,testB,labelTest=get_benchmark_data('models/deepPPIPaddedTest.txt')
	else:
		trainA,trainB,labelTrain=get_benchmark_data('models/deepProfPPIPaddedTrain.txt')
		testA,testB,labelTest=get_benchmark_data('models/deepProfPPIPaddedTest.txt')

	conjointTrainA,conjointTrainB,ctdTrainA,ctdTrainB=other_descriptors_using_pydpi(trainA,trainB)
	conjointTestA,conjointTestB,ctdTestA,ctdTestB=other_descriptors_using_pydpi(testA,testB)
	trainA,trainB=ac_encode(trainA,trainB)
	testA,testB=ac_encode(testA,testB)

	if dataToLoad==1:
		write_to_txt('data/descriptors/trainNoFilterAC.txt',protTrainA,protTrainB,trainA,trainB,labelTrain)
		write_to_txt('data/descriptors/testNoFilterAC.txt',protTestA,protTestB,testA,testB,labelTest)
	elif dataToLoad==2:
		write_to_txt('data/descriptors/SuppPaddedTrainAC.txt',protTrainA,protTrainB,trainA,trainB,labelTrain)
		write_to_txt('data/descriptors/SuppPaddedTestAC.txt',protTestA,protTestB,testA,testB,labelTest)
	elif dataToLoad==5:
		write_to_txt('data/descriptors/trainNoFilterPhysicalAC.txt',protTrainA,protTrainB,trainA,trainB,labelTrain)
		write_to_txt('data/descriptors/testNoFilterPhysicalAC.txt',protTestA,protTestB,testA,testB,labelTest)
	elif dataToLoad==6:
		write_to_txt('data/descriptors/trainNoFilterPhysicalAllPosAC.txt',protTrainA,protTrainB,trainA,trainB,labelTrain)
		write_to_txt('data/descriptors/testNoFilterPhysicalAllPosAC.txt',protTestA,protTestB,testA,testB,labelTest)
		write_to_txt('data/descriptors/trainNoFilterPhysicalAllPosCTD.txt',protTrainA,protTrainB,ctdTrainA,ctdTrainB,labelTrain)
		write_to_txt('data/descriptors/testNoFilterPhysicalAllPosCTD.txt',protTestA,protTestB,ctdTestA,ctdTestB,labelTest)
		write_to_txt('data/descriptors/trainNoFilterPhysicalAllPosConjoint.txt',protTrainA,protTrainB,conjointTrainA,conjointTrainB,labelTrain)
		write_to_txt('data/descriptors/testNoFilterPhysicalAllPosConjoint.txt',protTestA,protTestB,conjointTestA,conjointTestB,labelTest)
	elif dataToLoad==7:
		write_to_txt_benchmark('data/descriptors/deepPPIPaddedTrainAC.txt',trainA,trainB,labelTrain)
		write_to_txt_benchmark('data/descriptors/deepPPIPaddedTestAC.txt',testA,testB,labelTest)
		write_to_txt_benchmark('data/descriptors/deepPPIPaddedTrainCTD.txt',ctdTrainA,ctdTrainB,labelTrain)
		write_to_txt_benchmark('data/descriptors/deepPPIPaddedTestCTD.txt',ctdTestA,ctdTestB,labelTest)
		write_to_txt_benchmark('data/descriptors/deepPPIPaddedTrainConjoint.txt',conjointTrainA,conjointTrainB,labelTrain)
		write_to_txt_benchmark('data/descriptors/deepPPIPaddedTestConjoint.txt',conjointTestA,conjointTestB,labelTest)
	else:
		write_to_txt('data/descriptors/deepProfPPIPaddedTrainAC.txt',protTrainA,protTrainB,trainA,trainB,labelTrain)
		write_to_txt('data/descriptors/deepProfPPIPaddedTestAC.txt',protTestA,protTestB,testA,testB,labelTest)

	