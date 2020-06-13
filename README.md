# Using a Novel Unbiased Dataset and Deep Learning Model to Predict Protein-Protein Interactions

## Work of the last few weeks
### **Dataset:** 
Lately I've been focusing on two different datasets:

-**Dataset1**: from a benchmark, (http://www.csbio.sjtu.edu.cn/bioinf/LR_PPI/Data.htm).
-**Dataset2**: created by me, using BioGrid to obtain positive interactions, where two proteins interact. As for the negative interactions, they were created by random combinations of two proteins, in case this pair does not exist in all the positive interactions considered.


![negative](https://user-images.githubusercontent.com/58522514/78459150-f7ae3e00-76ae-11ea-87e7-6fbd41c81a53.PNG)

### **Feature encoding:**
As for the feature encoding, I kept the approach already used, eliminating interactions where at least one of the proteins had an excessively long sequence (there were cases of proteins with about 30,000 amino acids). Subsequently, I applied one-hot to each protein taking into account the existence of such 21 amino acids, thus obtaining matrices of 21xN. I opted for the representation of the 21 amino acids since I found that not grouping them in groups of 7, depending on their physical and chemical properties, allows to obtain better results.

### **Network architecture:** 
Lately I have focused on only one architecture, the most conventional, which was the one that for previous tests had shown better results.

![image](https://user-images.githubusercontent.com/58522514/78458093-c5004780-76a6-11ea-9d83-225c6f55310e.png)

### **Hyper-tuning:**
I focused my approach in adjusting some parameters:
- number of convolution layer filters, choosing random combinations of 3 of the following values [64,96,128,160,192,224,256,512];
- filter size, varying between 2 and 3;
- size of the pooling filter, varying between 2 and 3;
- number of neurons in the fully connected layers, choosing random combinations of 3 of the following values [64,128,256,512,1024];
- drop rate of the dropout layers that were interspersed with the fully connected layers, their value being 0.1, 0.2 or 0.3;
- activation function, choosing 'relu' or 'swish';
- model optimizers, opting for Adam, SGD, or even RMSprop, as well as the learning rate of these same optimizers, choosing between 0.01 and 0.001.

### **Results:** 
Dividing for each of the datasets:
-**Dataset1**: Since there are several articles that choose to apply the same dataset, I chose to build a table where you can see the results of the different models proposed. For this dataset the best model has the following characteristics:
   - number of filters in the convolution layers: [96,128,256];
   - filter size: 2;
   - size of the pooling filter: 2;
   - number of neurons in the fully connected layers, [64,256,1024];
   - drop rate of the dropout layers: 0.1;
   - activation function: 'relu';
   - model optimizers: Adam
   - Learning Rate: 0.001.
  
Model  | Accuracy (%)| 
----------- | -------------
https://pubs.acs.org/doi/abs/10.1021/pr100618t | 97.9
https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/s12859-017-1700-2 | 96.8
https://www.sciencedirect.com/science/article/abs/pii/S0025556418307168 | 98.3
My model  | 98.4


- **Dataset2**: For this dataset the best model obtained an accuracy of 67% and has the following characteristics:
   - number of convolution layer filters: [64,224,128];
   - filter size: 2;
   - size of the pooling filter: 2;
   - number of neurons in the fully connected layers, [256,64,128];
   - drop rate of the dropout layers: 0.1;
   - activation function: 'relu';
   - model optimizers: Adam
   - Learning Rate: 0.001.

### **Future work:** 
In a way, this comparison work, using datasets already published, was fundamental to validate the model and reach the conclusion that to obtain results in the range of excellent, the dataset is fundamental. In the future I had thought of exploring:
- dataset of two benchmarks (https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00028 and https://www.ncbi.nlm.nih.gov/pubmed/25657331) and evaluate their results to validate the model;
- dataset created by me where only the multi-validated physical interactions of Biogrid are considered as positive, with the negative interactions being created by the same process, previously explained.
- evaluate the efficiency and decision-making capacity of 3 different architectures:
  - architecture based on the implementation of convolutional layers followed by an LSTM layer that function as an extractor of features of each protein, being subsequently concatenated into a feature vector, specific to each interaction and evaluated and classified as an interaction pair or not;
  - architecture that encompasses different ways of representing each of the proteins, where for example each protein is represented by different descriptors, such as conjoint triads, auto-covariance, among others. The main focus being to evaluate whether the different information from different representations can complement each other and thereby create a model with better generalization capacity;
  - architecture where only convolutional layers are used, where downsampling, typically done by pooling layers,
is now achieved by using different stride values.

