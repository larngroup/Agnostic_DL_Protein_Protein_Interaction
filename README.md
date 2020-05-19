# Thesis

## Work of the last few weeks
### **Dataset:** 
Ultimamente tenho-me focado em dois datasets diferentes:
- **Dataset1**: de um benchmark, (http://www.csbio.sjtu.edu.cn/bioinf/LR_PPI/Data.htm).
- **Dataset2**: criado por mim, com recurso ao BioGrid para obter as interações positivas, onde duas proteínas interagem. Quanto às interações negativas, foram criadas por combinações aleatórias de duas proteínas, caso esse par não exista em todas as interações positivas consideradas.

![negative](https://user-images.githubusercontent.com/58522514/78459150-f7ae3e00-76ae-11ea-87e7-6fbd41c81a53.PNG)

### **Feature encoding:**
Quanto ao feature encoding, mantive a abordagem já usada, eliminei interações onde pelo menos uma das proteínas tivesse uma sequência excessivamente grande (existiam casos de proteínas com cerca de 30000 aminoácidos). Posteriormente, a cada proteína apliquei one-hot tendo em conta a existência dos tais 21 aminoácidos, obtendo assim matrizes de 21xN. Optei pela representação dos 21 aminoácidos uma vez que verifiquei que não agrupar estes mesmos em grupos de 7, consoantes as suas propriedades físico-químicas, permite obter melhores resultados.

### **Network architecture:** 
Ultimamente foquei-me apenas numa arquitectura, a mais convencional, que foi aquela que para os testes anteriores tinha apresentado melhores resultados.

![image](https://user-images.githubusercontent.com/58522514/78458093-c5004780-76a6-11ea-9d83-225c6f55310e.png)

### **Hyper-tuning:**
Centrei a aminha abordagem no ajuste de alguns parâmetros, passando desde já a enumerá-los:
- número de filtros das camadas de convolução, escolhendo combinações aleatórias de 3 dos seguintes valores [64,96,128,160,192,224,256,512];
- tamanho do filtro, variando entre 2 e 3;
- tamanho do filtro de pooling, variando entre 2 e 3;
- número de neurónios das camadas fully connected, escolhendo combinações aleatórias de 3 dos seguintes valores [64,128,256,512,1024];
- drop rate das camadas de dropout que foram intercaladas com as camadas fully connected, sendo o seu valor 0.1, 0.2 ou 0.3;
- função de ativação, optando por 'relu' ou 'swish';
- otimizadores do modelo, optando por Adam, SGD, ou ainda RMSprop, assim como o learning Rate destes mesmo otimizadores, escolhendo entre 0.01 e 0.001.

### **Results:** 
Dividindo para cada um dos datasets:
- **Dataset1**: Uma vez que existem diversos artigos que optam por aplicar o mesmo dataset, optei por construir uma tabela onde se pode observar os resultados dos diferentes modelos propostos. Para este dataset o melhor modelo apresenta as seguintes características:
  - número de filtros das camadas de convolução:[96,128,256];
  - tamanho do filtro: 2;
  - tamanho do filtro de pooling: 2;
  - número de neurónios das camadas fully connected, [64,256,1024];
  - drop rate das camadas de dropout: 0.1;
  - função de ativação: 'relu';
  - otimizadores do modelo: Adam
  - learning Rate: 0.001.
  
Model  | Accuracy (%)| 
----------- | -------------
https://pubs.acs.org/doi/abs/10.1021/pr100618t | 97.9
https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/s12859-017-1700-2 | 96.8
https://www.sciencedirect.com/science/article/abs/pii/S0025556418307168 | 98.3
Meu modelo  | 98.4


- **Dataset2**: Para este dataset o melhor modelo permitiu obter uma accuracy de 67% apresenta as seguintes características:
  - número de filtros das camadas de convolução:[64,224,128];
  - tamanho do filtro: 2;
  - tamanho do filtro de pooling: 2;
  - número de neurónios das camadas fully connected, [256,64,128];
  - drop rate das camadas de dropout: 0.1;
  - função de ativação: 'relu';
  - otimizadores do modelo: Adam
  - learning Rate: 0.001.

### **Future work:** 
De certa forma este trabalho de comparação, utilizando datasets já publicados, penso que deu para validar o modelo e chegar à conclusão que para obter resultados na gama do excelente, o dataset é fundamental. Futuramente tinha pensado em explorar implementar:
- dataset de dois benchmarks (https://pubs.acs.org/doi/abs/10.1021/acs.jcim.7b00028 e https://www.ncbi.nlm.nih.gov/pubmed/25657331) e avaliar os seus resultados para validar o modelo;
- dataset criado por mim onde apenas são consideradas como positivas as interações físicas multi-validated do Biogrid, sendo as interações negativas criadas pelo mesmo processo, já explicado anteriormente.
- avaliar a eficiência e capacidade de decisão de 3 diferentes arquitecturas:
  - arquitetura baseada na implementação de camadas convolucionais seguidas de uma camada LSTM que funcionam como extrator de features de cada proteína, sendo posteriormente concatenadas num vetor de features, específico para cada interação e avaliados e classificados como um par de interação ou não;
  - arquitetura que engloba diversas maneiras de representação de cada uma das proteínas, onde por exemplo cada protéina é representada por diferentes descritores, como conjoint triads, auto-covariance, entre outros. Sendo o principal foco avaliar se as diferentes informações provenientes das diferentes representações se podem complementar e com isso criar um modelo com melhor capacidade de generalização;
  - arquitetura onde apenas são utilizadas camadas convolucionais, onde o downsampling, tipicamente feito pelas camadas de pooling, 
é agora atingido pela utilização de diferentes valores de stride.

