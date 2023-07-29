import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

sns.set(style='darkgrid')

X=np.random.random((100,50))-0.5
Y=np.where(X.mean(axis=1)>0 ,1,0)

def initialize_model(activation ,learning_rate):
    model =tf.keras.models.Sequential([
        tf.keras.layers.Dense(4,input_shape =(50,),activation =activation),
        tf.kras.layers.Dense(1,activation = 'sigmoid')
    ])
    
    model.complie(optimizer = tf.keras.optimizer.SGD(lr = learning_rate),
                  loss ='binary_crossentropy'
                )
    
    return model

sigmoid_model = initialize_model('sigmoid', 0.1)
tanh_model = initialize_model('tanh', 0.1)

sigmoid_res = sigmoid_model.fit(X,Y , epochs =50 , verbose =0)
print("Last loss of sigmoid model:", sigmoid_res.history['loss'][-1])


tanh_res = tanh_model.fit(X,Y , epochs=50 , verbose= 0)
print("Last loss of sigmoid model:", tanh_res.history['loss'][-1])


final_loss ={'tanh' : [] , 'sigmoid' : []}
n_epochs =30

learning_rates =np.arange(0.1,8,0.3)
for learning_rate in learning_rates:
    sigmoid_model = initialize_model('sigmoid' , learning_rate)
    tanh_model = initialize_model('tanh' , learning_rate)
    
    sigmoid_res = sigmoid_model.fit(X, Y, epochs=n_epochs, verbose=0)
    final_loss['sigmoid'].append(sigmoid_res.history['loss'][-1])
    
    tanh_res = tanh_model.fit(X, Y, epochs=n_epochs, verbose=0)
    final_loss['tanh'].append(tanh_res.history['loss'][-1])
    

final_loss_df = pd.DataFrame(final_loss, index=learning_rates)
final_loss_df.plot(figsize=(10,5))