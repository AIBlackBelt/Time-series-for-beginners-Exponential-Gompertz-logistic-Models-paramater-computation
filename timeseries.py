from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import math

#### importing data ####
# Load dataset
url = "/home/dark/Mywork/time serie prediction for beginners/Data.csv"
#url = "E:\ENSIAS\Courses\IA\TPs\IMDB Dataset.csv"
dataset = read_csv(url)

#### data partitioning ####


def partition_data(data):
    T =  data.shape[0]
    if T%3 == 0:
        first_subset = np.array([data[i] for i in range(0,int(T//3))])
        second_subset = np.array([data[i] for i in range(int(T//3),int(2*(T//3)))])
        third_subset = np.array([data[i] for i in range(2*(T//3),T)])
    if (T-1)%3 == 0:
        first_subset = np.array([data[i] for i in range(0,int(T//3))])
        second_subset = np.array([data[i] for i in range(int(T//3),int(T//3+(T-1)/3 + 1))])
        third_subset = np.array([data[i] for i in range(int(T//3+(T-1)/3 + 1),T)])
    if (T-2)%3 == 0:    
        first_subset = np.array([data[i] for i in range(0,int((T-2)/3 + 1))])
        second_subset = np.array([data[i] for i in range(int((T-2)/3 + 1),int((T-2)/3 + 1+(T-2)/3))])
        third_subset = np.array([data[i] for i in range(int((T-2)/3 + 1+(T-2)/3),T)])
    
    return [first_subset,second_subset,third_subset]

def compute_t1_t2_t3_y1_y2_3(partition):
     T = [0]*3
     Y = [0]*3
     for k in range(0,len(partition[0])):
         Y[0] = Y[0] + partition[0][k]
         T[0] = T[0] + 1 
     save_time = T[0]
     Y[0] = Y[0]/len(partition[0])
     T[0] = T[0]/len(partition[0])
     T[1] = save_time
     for k in range(0,len(partition[1])):
         Y[1] = Y[1] + partition[1][k]
         T[1] = T[1] + 1 
     save_time =  T[1]
     Y[1] = Y[1]/len(partition[1])
     T[1] = T[1]/len(partition[1])
     T[2] = save_time
     for k in range(0,len(partition[2])):
         Y[2] = Y[2] + partition[2][k]
         T[2] = T[2] + 1 
     Y[2] = Y[2]/len(partition[2])
     T[2] = T[2]/len(partition[2])
     return [T,Y]

def compute_alpha_beta_gamma_exponential_model(T_Y_values):

   delta = T_Y_values[0][1] - T_Y_values[0][0]
   beta = math.exp((1/delta)*np.log(T_Y_values[1][2] - T_Y_values[1][1] /(T_Y_values[1][1] - T_Y_values[1][0])))
   alpha = (T_Y_values[1][2] - T_Y_values[1][1])/(math.exp(T_Y_values[0][2]*np.log(beta))-math.exp(T_Y_values[0][1]*np.log(beta)))
   gamma = T_Y_values[1][0] - alpha*math.exp(T_Y_values[0][0]*np.log(beta))

   return {"alpha parameter for exponential model": alpha,"beta parameter for exponential model" :beta,"gamma parameter for exponential model":gamma}

def compute_alpha_beta_gamma_Gompertz_model(T_Y_values):

   delta = T_Y_values[0][1] - T_Y_values[0][0]
   beta = math.exp((1/delta)*np.log((np.log(T_Y_values[1][2]) - np.log(T_Y_values[1][1])) /(np.log(T_Y_values[1][1]) - np.log(T_Y_values[1][0]))))
   alpha = (np.log(T_Y_values[1][2]) - np.log(T_Y_values[1][1]))/(math.exp(T_Y_values[0][2]*np.log(beta))-math.exp(T_Y_values[0][1]*np.log(beta)))
   gamma = T_Y_values[1][0] - alpha*math.exp(T_Y_values[0][0]*np.log(beta))

   return {"alpha parameter for Gompertz model": alpha,"beta parameter for Gompertz model" :beta,"gamma parameter for Gompertz model":gamma}


def compute_alpha_beta_gamma_logistic_model(T_Y_values):

   delta = T_Y_values[0][1] - T_Y_values[0][0]
   beta = math.exp((1/delta)*np.log(((1/T_Y_values[1][2]) - (1/T_Y_values[1][1])) /((1/T_Y_values[1][1]) - (1/T_Y_values[1][0]))))
   alpha = ((1/T_Y_values[1][2]) - (1/T_Y_values[1][1]))/(math.exp(T_Y_values[0][2]*np.log(beta))-math.exp(T_Y_values[0][1]*np.log(beta)))
   gamma = (1/T_Y_values[1][0]) - alpha*math.exp(T_Y_values[0][0]*np.log(beta))
   return {"alpha parameter for logistic model": alpha,"beta parameter for logistic model" :beta,"gamma parameter for logistic model":gamma}

def compute_logistic_model(alpha,beta,gamma,timestep):

    return 1/((alpha*math.exp(timestep*np.log(beta)))+gamma)
def compute_exponential_model(alpha,beta,gamma,timestep):

    return ((alpha*math.exp(timestep*np.log(beta)))+gamma)
def compute_Gompertz_model(alpha,beta,gamma,timestep):

    return math.exp((alpha*math.exp(timestep*np.log(beta)))+gamma)

def loss_function_MSE_Gompertz_model(data,model_parameters):
    S = 0
    for i in range(0,len(data)):
        S = S + (data[i] - compute_Gompertz_model(model_parameters["alpha parameter for Gompertz model"],model_parameters["beta parameter for Gompertz model"],model_parameters["gamma parameter for Gompertz model"],i+1))**2
    
    return S/len(data)

def loss_function_MSE_exponential_model(data,model_parameters):
    S = 0
    for i in range(0,len(data)):
        S = S + (data[i] - compute_exponential_model(model_parameters["alpha parameter for exponential model"],model_parameters["beta parameter for exponential model"],model_parameters["gamma parameter for exponential model"],i+1))**2
    
    return S/len(data)

def loss_function_MSE_logistic_model(data,model_parameters):
    S = 0
    for i in range(0,len(data)):
        S = S + (data[i] - compute_logistic_model(model_parameters["alpha parameter for logistic model"],model_parameters["beta parameter for logistic model"],model_parameters["gamma parameter for logistic model"],i+1))**2
    
    return S/len(data)



inputs = partition_data(np.array(dataset["Belgic"]))
outputs = compute_t1_t2_t3_y1_y2_3(inputs)
print("Belgic PIB data model :")
exponential_parameters=compute_alpha_beta_gamma_exponential_model(outputs)
gompertz_parameters=compute_alpha_beta_gamma_Gompertz_model(outputs)
logistic_parameters=compute_alpha_beta_gamma_logistic_model(outputs)
print("loss function for exponential model",loss_function_MSE_exponential_model(dataset["Belgic"],exponential_parameters))
print("loss function for Gompertz model",loss_function_MSE_Gompertz_model(dataset["Belgic"],gompertz_parameters))
print("loss function for logistic model",loss_function_MSE_logistic_model(dataset["Belgic"],logistic_parameters))


inputs = partition_data(np.array(dataset["RFA"]))
outputs = compute_t1_t2_t3_y1_y2_3(inputs)
print("RFA PIB data model :")
exponential_parameters=compute_alpha_beta_gamma_exponential_model(outputs)
gompertz_parameters=compute_alpha_beta_gamma_Gompertz_model(outputs)
logistic_parameters=compute_alpha_beta_gamma_logistic_model(outputs)
print("loss function for exponential model",loss_function_MSE_exponential_model(dataset["RFA"],exponential_parameters))
print("loss function for Gompertz model",loss_function_MSE_Gompertz_model(dataset["RFA"],gompertz_parameters))
print("loss function for logistic model",loss_function_MSE_logistic_model(dataset["RFA"],logistic_parameters))

