time series method to estimate : 
exponential trend y(t) = αlpha*(βetaraised_to(t)) + gamma 

logistic trend : y(t) = 1/(αlpha*(βetaraised_to(t)) + gamma)

Gromptz trend : y(t) = exp((αlpha*(βetaraised_to(t)) + gamma)) 

parameters alpha,beta,gamma, therefore each model will have its own alpha,beta,gamma.

suppose our sequence dataset presents T timestamps
1) sort data by chronological order into tree subsets I,II, III, of same size (approximately), to the point where duration average of TI ,TII et TIII could be as close to each other as possible :
(a) if T is multiple dof 3, all three subsets will be of the same size.
(b) if (T -1) is multiple of 3, put the subset of size (T-1)/3 + 1 in
II and the two other subsets of size T/3 in I et III
(c) if (T -2) is multiple of 3, put the subset of size (T-2)/3 in
II and the two other subsets of size (T -2)/3 + 1 in I et III

Calculer les médianes (ou les moyennes) de y t pour les trois sous-
ensembles : YI , YII et YIII



The data that we will be using to simulate : 
Year 
1972
1973
1974
1975
1976
1977
1978
1979

Belgic PIB
31.3 
36.7
44.4
49.8
59.6
68
74.4
79

8 observations

we are in case c : 

data repartition : 

the subset I will be composed of 3 observations, 31.3 36.7 44.4 

the subset II will be composed of 2 observations, 49.8 59.6 

the subset III will be composed of 3 obsevartion, 68 74.4 79

lets now compute the mean for all tree subsets : 
Subset I mean : 31.3 36.7 44.4, noted by YI = (31.3+36.7+44.4)/3 : 37.46
Subset II mean : 31.3 36.7 44.4, noted by YII =  (49.8+59.6)/2 : 54,7
Subset III mean : 68 74.4 79, noted by YIII = (68+74.4+79)/3 : 73,8


delta = TII - TI = TIII - TII = 4.5 - 2 =   7 - 4.5 = 2.5

You can follow the mathematical demonstration in the folder equation screenshots, the final expressions to estimate alpha,beta,gamma for exponential model are given in the screenshot figure 5,
Exponential model : 
Beta = Exp((1/delta)*ln((YIII - YII)/(YII - YI)))
alpha = (YIII - YII)/ (Betaraised_to(TIII) - Betaraised_to(TII))
gamma = YI - alpha*Betaraised_to(TI)

Gompertz model : 
We can estimate these model parameter by using the previous expressions, in fact both models are equivalent if we consider Y(exponential model) = 1/Y(Gompertz model) 

logistic model : 
We can estimate these model parameter by using the previous expressions, in fact both models are equivalent if we consider Y(exponential model) = ln(Y( logistic model) 

We provide the loss function for all of these models applied for each dataset (we apply these models on two datasets belgic and RFA PIB)

