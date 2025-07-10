import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('dataset/train.csv')
Y = df.filter(like="Blend")
X = df.drop(columns=Y.columns, axis=1)

X_train, X_val, Y_train, Y_val = ms.train_test_split(X, Y, test_size=0.2, random_state=42)

model= ([
    Dense(int(X_train.shape[1]),),
    Dense(128,activation='relu'),
    Dense(Y_train.shape[1])
])


model.compile(optimizer='adam',loss='mse',metrics=['mae'])

model.fit(X_train,Y_train,epochs=50,batch_size=32,validation_data=(X_val,Y_val))

model_predictions = model.predict(X_val)

# calculate the mean absolute percentage error (MAPE)
mape = np.mean(np.abs((Y_val - model_predictions) / Y_val)) * 100
print("Mean Absolute Percentage Error (MAPE):", mape)

ref1 = 2.72
ref = 2.73
def leader_board_score(mape, x):
    return max(10,(100 -((90 *mape )/x)))

print("Leader Board Score:", leader_board_score(mape, ref1))        
print("Leader Board Score:", leader_board_score(mape, ref))