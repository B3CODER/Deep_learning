import numpy as np
# [5,4,3]: There were two inputs, one hidden layer with 4 hidden units, and an output layer 
# with 3 output unit. This means W1's shape was (4,5), b1 was (4,1), W2 was (3,4) and b2 was (3,1). 
# Now you will generalize this to layers!

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters={}
    L = len(layer_dims)
    
    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])
        parameters['b'+str(l)] = np.zeros((layer_dims[l],1))
        
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

parameters = initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s
def sigmoid_backward(dA, cache):

    Z = cache 
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ
    
def relu(x):
    return max(0.0,x)

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear_forward(A,W,b):
    Z=np.dot(W,A)+b
    assert(Z.shape ==(W.shape[0],A.shape[1]))
    cache =(A,W,b)
    return Z,cache

def linear_activation_function_forward(A_prev,W,b,activation):
    
    if(activation == 'sigmoid'):
        Z,linear_cache =linear_forward(A_prev ,W,b)
        A , activation_cache =sigmoid(Z)
        
    elif(activation=="relu"):
        Z,linear_cache =linear_forward(A_prev,W,b)
        A,activation_cache =relu(Z)
        

    assert (A.shape == (W.shape[0],A.prev.shape[1]))
    cache =(linear_cache,activation_cache)
    
    return A,cache

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)

        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db
# L layer model
def L_model_forward(X,parameters):
    caches =[]
    A=X
    L=len(parameters) //2 
    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_function_forward(A_prev ,parameters['W'+str(l)] ,parameters['b'+str(l)])
        caches.append(cache)
        
    
    AL,cache = linear_activation_function_forward(A,parameters['W'+str(L)],parameters['b'+str(L)])
    caches.append(cache)
    
    assert(AL.shape ==(1,X.shape[1]))
    
    return AL , caches

# GRADED FUNCTION: compute_cost

def compute_cost(AL, Y):
  
    m = Y.shape[1]

    cost = (-1/m) * (np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T))

    cost = np.squeeze(cost)      
    assert(cost.shape == ())
    
    return cost

def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1] # Last Layer
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    for l in reversed(range(L-1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters