import numpy as np
def forward_propagation(x,theta):
    J = np.dot(theta,x)
    return J
def backward_propagation(x,theta):
    dtheta =x
    return dtheta

def gradient_check(x,theta,epsilon=1e-7):
    theta_plus = theta + epsilon
    theta_minus = theta - epsilon
    J_plus =np.dot(theta_plus,x)
    J_minus = np.dot(theta_minus,x)
    gradapprox = (J_plus-J_minus)/(2*epsilon)
    
    grad =x
    
    numerator = np.linalg.norm(gradapprox-grad)                         # Step 1'
    denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)     # Step 2'
    difference = numerator/denominator 
    
    
    if difference < 1e-7:
        print ("The gradient is correct!")
    else:
        print ("The gradient is wrong!")
    
    return difference

x, theta = 2, 4
difference = gradient_check(x, theta)
print("difference = " + str(difference))