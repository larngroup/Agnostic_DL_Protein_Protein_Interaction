import os
from numpy import array
import numpy as np
import random

class DataCreator:

	def __init__(self, filtersList, filterGO, interactionType):
		self.filtersList=filtersList
		self.dics=[]
		self.prots=[]
		# using the GO filter, filterGO=1, if not, filterGO=2
		self.filterGO=filterGO
		# interactiontype=1 consider both physical and genetic interactions, if interactiontype=2 consider only physical
		# the only physical interactions are multi-validated (https://wiki.thebiogrid.org/doku.php/biogrid_mv)
		self.interactionType=interactionType
		self.get_positive_data(self.filtersList)


	def load_file(self,textFile, split):
		file=open('files/'+textFile, 'r').read().split(split)
		file=file[:len(file)-1]
		return file

	# Create a file with the unique proteins biogridID to posteriorly mapp
	def get_data_for_mapping(self,textFile):
		file=self.load_file(textFile,'\n')[1:]
		# Collects the BiogridID of protA and protB in the file
		A=[i.split('\t')[3] for i in file if i.split('\t')[15]=='9606']
		B=[i.split('\t')[4] for i in file if i.split('\t')[16]=='9606']

		ids=list(set(A+B))
		print('\nUnique proteins: ', len(ids))

		# Creates a file containing only the unique proteins to posteriorly mapp
		if self.interactionType==1:
			os.makedirs(os.path.dirname('files/mappingTable.txt'), exist_ok=True)
			with open('files/mappingTable.txt', 'w') as file:
				for a in ids:
					file.write(a+'\n')
		else:
			os.makedirs(os.path.dirname('files/mappingTablePhysical.txt'), exist_ok=True)
			with open('files/mappingTablePhysical.txt', 'w') as file:
				for a in ids:
					file.write(a+'\n')


	# Create a file where in col1 is the uniprotID of the interactorA and in col2 the uniprotID of interactorB
	def get_inters_by_uniprotIds(self,textFile, mappedFile):
		file1=self.load_file(textFile,'\n')[1:]
		file2=self.load_file(mappedFile,'\n')[1:]
		print('\nMapped proteins: ', len(file2))

		A=[i.split('\t')[3] for i in file1]
		B=[i.split('\t')[4] for i in file1]

		# When creating the mapping dictionary, BioID:UniProtID, 
		# takes in consideration that in some cases several BioID corresponds to 1 UniProtID
		dic={j:i.split('\t')[1] for i in file2 for j in i.split('\t')[0].split(',')}
		print('\nNumber of interactions in the original file: ',len(file1))

		# Removes interactions where at least one of the prots was not mapped
		indexInters=[i for i in range(len(A)) if A[i] in dic and B[i] in dic]
		UniA=[dic[A[i]] for i in indexInters]
		UniB=[dic[B[i]] for i in indexInters]
		print('\nNumber of interactions after removing the ones not mapped: ',len(UniA))

		if self.interactionType==1:
			os.makedirs(os.path.dirname('files/InteractionsByUniprot.txt'), exist_ok=True)
			with open('files/InteractionsByUniprot.txt', 'w') as file:
				for a,b in zip(UniA,UniB):
					file.write(a+'\t'+b+'\n')
		else:
			os.makedirs(os.path.dirname('files/InteractionsByUniprotPhysical.txt'), exist_ok=True)
			with open('files/InteractionsByUniprotPhysical.txt', 'w') as file:
				for a,b in zip(UniA,UniB):
					file.write(a+'\t'+b+'\n')

	# Create a list of dictionaries with key the uniprotID and value the sequence, where each dictionary corresponds to the file of specific GO
	def seq_dict(self,listFiles):
		listDic=[]
		rareAmino=['U','O','X','Z']
		for file in listFiles:
			data=self.load_file(file,'>')[1:]
			info=[prot.split('\n',1)[0] for prot in data if len(prot.split('\n',1))>1]
			seqs=[prot.split('\n',1)[1] for prot in data if len(prot.split('\n',1))>1]

			# Remore from the dic protein with rare amino and small proteins
			indToRemove=[i for i in range(len(seqs)) if any(amino in seqs[i] for amino in rareAmino) or len(seqs[i])<=40 ]
			for index in sorted(indToRemove, reverse=True):
				del seqs[index]
				del info[index]

			# Dic structure UniProtID:sequence
			dic={inf[inf.find('|')+1:inf.find('|',inf.find('|')+1)]: seq.replace('\n','') for inf, seq in zip(info,seqs)}
			listDic.append(dic)
		return listDic

	# Create 2 lists, one for each interactor, of all the interactions
	def get_inter(self,data):
		if len(data[0].split('\t'))==2:
			protA=[prot.split('\t')[0] for prot in data]
			protB=[prot.split('\t')[1] for prot in data]
		else:
			print('Ficheiro Stor')
			protA=[prot.split(' ')[1] for prot in data if prot.split(' ')[0]=='9606']
			protB=[prot.split(' ')[2] for prot in data if prot.split(' ')[0]=='9606']
		return protA, protB

	def write_to_txt(self,fileName,seqA,seqB, label):
		os.makedirs(os.path.dirname(fileName), exist_ok=True)
		with open(fileName, 'w') as file:
			for a,b in zip(seqA,seqB):
				file.write(str(a)+'\t'+str(b)+'\t'+str(label)+'\n')

	# function to filter interactions where one prot belongs to a specific GO and the other prot to another specific GO 
	def filtering_data_by_GO(self,dics,protA,protB):
		index=[]
		for i in range(0,len(dics),2):
			# Garantees that one prot belongs to dicA, according to a certain GO, and the other protein belongs to dicB
			indexInDic=[j for j in range(len(protA)) if (protA[j] in dics[i] and protB[j] in dics[i+1]) or (protB[j] in dics[i] and protA[j] in dics[i+1])]
			index=index+indexInDic

		# removes the same index of interactions if it exist
		index=list(set(index))
		protA=[protA[i] for i in index]
		protB=[protB[i] for i in index]

		return protA,protB

	def get_fraction_of_interactions(self,size, A, B):

		# Removes the interactions where one of the proteins do not exist in the dic, for example proteins with rare amino
		index=[i for i in range(len(A)) if (A[i] in self.dics[0] and B[i] in self.dics[0])]
		protA=[A[i] for i in index]
		protB=[B[i] for i in index]
		print("\nNumber of interactions in the file: "+str(len(protA)))
	
		index=random.sample([i for i in range(len(protA))], size)
		# while len(index)<size:
		# 	i=random.randint(0,len(protA)-1)
		# 	if i not in index:index.append(i)
		protA=[protA[i] for i in index]
		protB=[protB[i] for i in index]

		return protA,protB



	def get_positive_data(self,listOfFilesToFilter):

		# Verifies if the mapping was already done, otherwise does it
		if self.interactionType==1:
			if os.path.isfile('files/InteractionsByUniprot.txt')==False:
				self.get_data_for_mapping('BIOGRID-ORGANISM-Homo_sapiens-3.5.178.tab2.txt')
				self.get_inters_by_uniprotIds('BIOGRID-ORGANISM-Homo_sapiens-3.5.178.tab2.txt','mappedTable.tab')
		else:
			if os.path.isfile('files/InteractionsByUniprotPhysical.txt')==False:
				self.get_data_for_mapping('BIOGRID-MV-Physical-3.5.182.tab2.txt')
				self.get_inters_by_uniprotIds('BIOGRID-MV-Physical-3.5.182.tab2.txt','mappedTablePhysical.tab')

		# Use data given by teacher 
		# file1=self.load_file('InteractionsByUniprotTeacher.txt','\n')
		# Use my data
		if self.interactionType==1:
			file1=self.load_file('InteractionsByUniprot.txt','\n')
		else:
			file1=self.load_file('InteractionsByUniprotPhysical.txt','\n')

		dics=self.seq_dict(listOfFilesToFilter)
		self.dics=dics
		protA,protB=self.get_inter(file1)

		# Saving all the positive interactions. These will be used when creating the negative interactions
		# self.allPosA,self.allPosB=self.get_inter(self.load_file('InteractionsByUniprot.txt','\n'))
		self.allPosA=protA
		self.allPosB=protB	
		if self.filterGO==1:
			protA,protB=self.filtering_data_by_GO(self.dics,protA,protB)
			print('\nNumber of interactions after filtering only some type of interactions: ',len(protA))
		else:
			protA,protB=self.get_fraction_of_interactions(150000, protA, protB)
			print('\nNumber of interactions after removing some: ',len(protA))


		# Remove redundants interactions
		newA=[]
		newB=[]
		for a,b in zip(protA,protB):
			state=True
			# if the interaction or its inverse already exists in the new lists, then it wont be added
			for i in range(len(newA)):
				if a==newB[i] and b==newA[i]:
					state=False
					break
				elif a==newA[i] and b==newB[i]:
					state=False
					break
			if state==True:
				newA.append(a)
				newB.append(b)

		print('\nNumber of interactions after removing the redundant ones: ',len(newA))
		if self.interactionType==1:	
			if self.filterGO==1:
				self.write_to_txt('data/positiveDataFiltered.txt',newA,newB,1)
			else:
				self.write_to_txt('data/positiveDataNoFilter.txt',newA,newB,1)
		else:
			if self.filterGO==1:
				self.write_to_txt('data/positiveDataFilteredPhysical.txt',newA,newB,1)
			else:
				self.write_to_txt('data/positiveDataNoFilterPhysical.txt',newA,newB,1)
		# print(newA,newB)

# a=DataCreator(['chromatinbinding0003682.fasta','organiccycliccompoundbinding0097159.fasta'])