import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection , preprocessing
from sklearn.linear_model import LinearRegression ,Ridge ,Lasso ,RidgeCV ,LassoCV
from sklearn.metrics import mean_squared_error

# Generating data
np.random.seed(42)
x = np.sort(np.random.rand(100))
y = np.cos(1.2 * x * np.pi) + (0.1 * np.random.randn(100))

X_train , X_test , Y_train , Y_test = model_selection.train_test_split(x,y,test_size=0.2)

def Ridge_reg(lamda):
    x_train = X_train.reshape(-1,1)
    transf =preprocessing.PolynomialFeatures(degree=20)
    x_train =transf.fit_transform(x_train)
    clf=Ridge(alpha=lamda)
    clf.fit(x_train ,Y_train)
    train_accuracy = clf.score(x_train, Y_train)
    
    intercept = clf.intercept_
    coefficient = clf.coef_
    parameters = coefficient + intercept  #(y=c+mx)
    x_test = X_test.reshape(-1,1)
    transf = preprocessing.PolynomialFeatures(degree = 20)
    x_test = transf.fit_transform(x_test)
    test_accuracy = clf.score(x_test, Y_test)
    train_predict = clf.predict(x_train)
    train_MSE = mean_squared_error(Y_train, train_predict)
    test_predict = clf.predict(x_test)
    test_MSE = mean_squared_error(Y_test, test_predict)
    print('Train accuracy:', train_accuracy, '\n')
    print('Test accuracy:', test_accuracy, '\n')
    print('Train MSE', train_MSE, '\n')
    print('Test MSE', test_MSE, '\n')
    print('Parameters:', parameters)
    
    x_model = x.reshape(-1,1)
    x_model = transf.fit_transform(x_model)
    y_model = clf.predict(x_model)
    x_test = X_test
    
    
    fig =plt.figure(figsize=(20,10))
    ax1 =fig.add_subplot(1,2,1)
    ax2 =fig.add_subplot(1,2,2)
    ax1.scatter(X_train,Y_train , color ='k' , label ="Training examples ")
    ax1.plot(x,np.cos(1.2*x*np.pi), linewidth =3 ,label ='True function')
    ax1.plot(x,y_model  ,label='Model function' , linewidth =3)
    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax2.scatter(X_test, Y_test, color = 'r', label = 'Testing examples')
    ax2.plot(x, np.cos(1.2 * x * np.pi), linewidth = 3, label = 'True function')
    ax2.scatter(X_test, test_predict,color = 'k', label = 'Model predictions')
    plt.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.show()
    
Ridge_reg(0)
# We can see that as the value of lambda increases the model becomes a straight line parallel to x-axis.

