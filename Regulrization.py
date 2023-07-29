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

# Plotting the generated data
fig =plt.figure(figsize=(5,5))
sns.set(style = 'whitegrid')
plt.scatter(X_train ,Y_train , color ='k' , label ='Train data')
plt.scatter(X_test , Y_test ,color='r' , label ='Test data')
plt.plot(x,np.cos(1.2*x*np.pi) , linewidth =3 , label = "True-fit")
plt.xlabel('x')
plt.ylabel('y')
# plt.legend()
# plt.show()

# Using an equation of degree 1
x_train =X_train.reshape(-1,1)
clf = LinearRegression()
clf.fit(x_train , Y_train)
train_accuracy = clf.score(x_train, Y_train)
print('train accuracy', train_accuracy)


x_test = X_test.reshape(-1,1)
test_accuracy = clf.score(x_test, Y_test)
print('test accuracy', test_accuracy)

train_predict = clf.predict(x_train)
train_MSE = mean_squared_error(Y_train, train_predict)
print('Training MSE:', train_MSE)

test_predict = clf.predict(x_test)
test_MSE = mean_squared_error(Y_test, test_predict)
print('Test MSE:', test_MSE)


fig = plt.figure(figsize = (10,10))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.scatter(X_train, Y_train, color = 'k', label = 'Training examples')
ax1.plot(x, np.cos(1.2 * x * np.pi), linewidth = 3, label = 'True function')
ax1.plot(X_test, test_predict, label = 'Model predictions' )
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.scatter(X_test, Y_test, color = 'r', label = 'Testing examples')
ax2.plot(x, np.cos(1.2 * x * np.pi), linewidth = 3, label = 'True function')
ax2.scatter(X_test, test_predict,color = 'k', label = 'Model predictions')
plt.legend()
ax2.set_xlabel('x')
ax2.set_ylabel('y')
# plt.show()

print("\nUsing an equation of degree 2")


transf = preprocessing.PolynomialFeatures()
x_train = X_train.reshape(-1,1)
x_train = transf.fit_transform(x_train)
clf = LinearRegression()
clf.fit(x_train ,Y_train)
train_accuracy = clf.score(x_train, Y_train)
print('Train accuracy:', train_accuracy)

x_test = X_test.reshape(-1,1)
transf = preprocessing.PolynomialFeatures()
x_test = transf.fit_transform(x_test)
test_accuracy = clf.score(x_test, Y_test)
print('test accuracy', test_accuracy)

train_predict = clf.predict(x_train)
train_MSE = mean_squared_error(Y_train, train_predict)
print('Training MSE:', train_MSE)

test_predict = clf.predict(x_test)
test_MSE = mean_squared_error(Y_test, test_predict)
print('Test MSE:', test_MSE)

x_model = x.reshape(-1,1)
x_model = transf.fit_transform(x_model)
y_model = clf.predict(x_model)
x_test = X_test

fig = plt.figure(figsize = (10,10))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.scatter(X_train, Y_train, color = 'k', label = 'Training examples')
ax1.plot(x, np.cos(1.2 * x * np.pi), linewidth = 3, label = 'True function')
ax1.plot(x, y_model, label = 'Model function', linewidth = 3 )
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.scatter(X_test, Y_test, color = 'r', label = 'Testing examples')
ax2.plot(x, np.cos(1.2 * x * np.pi), linewidth = 3, label = 'True function')
ax2.scatter(X_test, test_predict,color = 'k', label = 'Model predictions')
plt.legend()
ax2.set_xlabel('x')
ax2.set_ylabel('y')
# plt.show()
 

print("\nUsing an equation of degree 20")


transf = preprocessing.PolynomialFeatures(20)
x_train = X_train.reshape(-1,1)
x_train = transf.fit_transform(x_train)
clf = LinearRegression()
clf.fit(x_train ,Y_train)
train_accuracy = clf.score(x_train, Y_train)
print('Train accuracy:', train_accuracy)

x_test = X_test.reshape(-1,1)
transf = preprocessing.PolynomialFeatures(20)
x_test = transf.fit_transform(x_test)
test_accuracy = clf.score(x_test, Y_test)
print('test accuracy', test_accuracy)

train_predict = clf.predict(x_train)
train_MSE = mean_squared_error(Y_train, train_predict)
print('Training MSE:', train_MSE)

test_predict = clf.predict(x_test)
test_MSE = mean_squared_error(Y_test, test_predict)
print('Test MSE:', test_MSE)

x_model = x.reshape(-1,1)
x_model = transf.fit_transform(x_model)
y_model = clf.predict(x_model)
x_test = X_test

fig = plt.figure(figsize = (10,10))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.scatter(X_train, Y_train, color = 'k', label = 'Training examples')
ax1.plot(x, np.cos(1.2 * x * np.pi), linewidth = 3, label = 'True function')
ax1.plot(x, y_model, label = 'Model function', linewidth = 3 )
ax1.legend()
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.scatter(X_test, Y_test, color = 'r', label = 'Testing examples')
ax2.plot(x, np.cos(1.2 * x * np.pi), linewidth = 3, label = 'True function')
ax2.scatter(X_test, test_predict,color = 'k', label = 'Model predictions')
plt.legend()
ax2.set_xlabel('x')
ax2.set_ylabel('y')
# plt.show()

# Checking MSE for different degrees
def Evaluation(degree):
    x_train =X_train.reshape(-1,1)
    transf = preprocessing.PolynomialFeatures(degree = degree)
    x_train = transf.fit_transform(x_train)
    clf = LinearRegression()
    clf.fit(x_train, Y_train)
    train_accuracy = clf.score(x_train, Y_train)
    x_test = X_test.reshape(-1,1)
    transf = preprocessing.PolynomialFeatures(degree = degree)
    x_test = transf.fit_transform(x_test)
    test_accuracy = clf.score(x_test, Y_test)
    train_predict = clf.predict(x_train)
    train_MSE = mean_squared_error(Y_train, train_predict)
    test_predict = clf.predict(x_test)
    test_MSE = mean_squared_error(Y_test, test_predict)
    return train_accuracy, test_accuracy, train_MSE, test_MSE
    
Train_accuracy = []
Test_accuracy = []
Train_MSE = []
Test_MSE = []
   
for i in range(40):
    a, b, c, d = Evaluation(i+1)
    Train_accuracy.append(a)
    Test_accuracy.append(b)
    Train_MSE.append(c)
    Test_MSE.append(d)
    
degrees = np.linspace(1, 40, 40)
fig = plt.figure(figsize = (20,10))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(degrees, Train_accuracy, label = 'Training accuracy', linewidth = 3)
ax1.plot(degrees, Test_accuracy, label = 'Testing accuracy', linewidth = 3)
ax1.legend()
ax1.set_xlabel('Degrees')
ax1.set_ylabel('Accuracy')
ax2.plot(degrees, Train_MSE, label = 'Training MSE', linewidth = 3)
ax2.plot(degrees, Test_MSE, label = 'Testing MSE', linewidth = 3)
plt.ylim(0, 0.05)
plt.legend()
ax2.set_xlabel('Degrees')
ax2.set_ylabel('MSE')
# plt.show()

Test_min_degree = Test_MSE.index(min(Test_MSE))+1
print("Min test error occur at " ,Test_min_degree)


x_train = X_train.reshape(-1,1)
transf = preprocessing.PolynomialFeatures(degree = 9)
x_train = transf.fit_transform(x_train)
clf = LinearRegression()
clf.fit(x_train, Y_train)
train_accuracy = clf.score(x_train, Y_train)
print('Train accuracy:', train_accuracy)

x_test = X_test.reshape(-1,1)
transf = preprocessing.PolynomialFeatures(degree = 9)
x_test = transf.fit_transform(x_test)
test_accuracy = clf.score(x_test, Y_test)
print('test accuracy', test_accuracy)