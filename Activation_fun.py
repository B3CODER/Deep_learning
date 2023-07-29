import numpy as np
import matplotlib.pyplot as plt
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A,Z

def tanh(Z):
    A =np.tanh(Z)
    return A,Z

def relu(Z):
    A = np.maximum(0,Z)
    return A,Z

def leaky_relu(Z):
    A = np.maximum(0.1*Z ,Z)
    return A,Z
# deifference between leaky relu and relu

z =np.linspace(-10,10,100)
A_sigmoid, z = sigmoid(z)
A_tanh, z = tanh(z)
A_relu, z = relu(z)
A_leaky_relu, z = leaky_relu(z)

#  plot sigmoid 
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(z, A_sigmoid, label="Function")
plt.plot(z, A_sigmoid * (1 - A_sigmoid), label = "Derivative") 
plt.legend(loc="upper left")
plt.title("Sigmoid Function", fontsize=10)

# plot tanh
plt.subplot(2,2,2)
plt.plot(z,A_tanh , 'b' , label ="Function")
plt.plot(z,1-np.square(A_tanh) , 'r' , label ='Derivative')
plt.legend(loc = "upper left")
plt.title("Tanh Function" , fontsize =10 )

# plot relu
plt.subplot(2,2,3)
plt.plot(z,A_relu , 'g')
plt.title("Relu Function" , fontsize =10)

# plot leaky relu
plt.subplot(2,2,4)
plt.plot(z,A_leaky_relu , 'g')
plt.title("Leaky Relu Function" , fontsize =10)
plt.tight_layout();

plt.show()

