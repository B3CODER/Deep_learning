import numpy as np
import math
def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s
x=np.array([1,2,3,4,5])
print(sigmoid(x))


def sigmoid_derivatives(x):
    s=sigmoid(x)
    ds = s*(1-s)
    return ds
print(sigmoid_derivatives(x))

# The loss is used to evaluate the performance of your model. The bigger your loss is, the more different your 
# predictions (y_hat) are from the true values (y). In deep learning, you use optimization algorithms like Gradient 
# Descent to train your model and to minimize the cost.

def L1(y_hat,y):
    loss = sum(abs(y_hat-y))
    return loss
yhat = np.array([0.9,.2,.1,.4,.9])
y=np.array([1,0,0,1,1])
print(L1(yhat,y))

def L2(yhat ,y):
    x = yhat -y
    loss = np.dot(x,x)
    return loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))
