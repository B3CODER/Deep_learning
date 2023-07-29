# only for concept
import numpy as np
def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L=len(layers_dims)
    
    for l in range(1,L):
        parameters['w' + str(l)] = np.zeros((layers_dims[l] ,layers_dims[l-1]))
        parameters['b' + str[l]] = np.zeros((layers_dims[l],1))
        
        
    return parameters

def intilizing_parameter_random(layers_dimes):
    parameters ={}
    L = len(layers_dimes)
    
    for l in range (1,L):
        parameters["w" + str(l)] = np.random.rand(layers_dimes[l],layers_dimes[l-1])*10
        parameters["b" + str(l)] = np.random.rand((layers_dimes[l],1))
        
    return parameters

def intilizing_parameters(layers_dims , intilization_parameter):
    parameters = {}
    L =len(layers_dims)
    
    if intilization_parameter == "he":
        for l in range(1,L):
            parameters["w" + str(l)] = np.random.rand(layers_dims[l] ,layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
            parameters["b" + str(l)] = np.zeros((layers_dims[l],1))
    elif intilization_parameter == "xavier":
        for l in range(1,L):
            parameters["w" + str(l)] = np.random.rand(layers_dims[l] ,layers_dims[l-1])*np.sqrt(1/layers_dims[l-1])
            parameters["b" + str(l)] = np.zeros((layers_dims[l],1))
            
    return parameters
    
    