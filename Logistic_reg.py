import numpy as np
def sigmoid(z):
    s =1/(1+np.exp(-z))
    return s
def Initializing_parameters(dim):
    w =np.zeros((dim,1))
    b=0
    return w,b
dim = 2
w, b = Initializing_parameters(dim)
print ("w = " + str(w))
print ("b = " + str(b))

def propagate(w,b,X,Y):
    m=X.shape[1]
    
    # forward propogation
    A = sigmoid(np.dot(w.T,X) + b)              # compute activation
    cost = np.sum(((- np.log(A))*Y + (-np.log(1-A))*(1-Y)))/m  # compute cost
    
    # backward propogation
    
    dw = (np.dot(X,(A-Y).T))/m
    db =  (np.sum(A-Y))/m
    

    
    grads = {"dw": dw, "db": db}
    
    return grads ,cost
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

#    ---->  Optimization

def optimize(w,b,X,Y,num_iterations, learning_rate , print_cost=False):
    costs =[]
    
    for i in range(num_iterations):
        grads ,cost =propagate(w,b,X,Y)
        
        dw =grads["dw"]
        db =grads["db"]
        
        
        w = w - (learning_rate*dw)
        b = b - (learning_rate*db)
        
        if i%100 ==0:
            costs.append(cost)
            
        if print_cost and i%100 ==0:
            print ("Cost after iteration %i: %f" %(i, cost))
        
        
        params = {"w": w,
              "b": b}
    
        grads = {"dw": dw,
             "db": db}
    
        return params, grads, costs
        
params, grads, costs = optimize(w, b, X, Y, num_iterations= 1000, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))


def predict(w,b,X):
    m =X.shape[1]
    Y_prediction = np.zeros((1,m))
    w =w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X) + b)
    
    Y_prediction = (A >= 0.5) * 1.0
    
    return Y_prediction
w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))
