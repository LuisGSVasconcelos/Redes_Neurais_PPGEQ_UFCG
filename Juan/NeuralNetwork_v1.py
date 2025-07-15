# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

df = pd.read_csv('banco_para_treinamento.csv');
df = df.astype('float16');

train_set, test_set = train_test_split(df, test_size=0.1);

train_set.reset_index(drop = True, inplace = True);
test_set.reset_index(drop = True, inplace = True);

Inputs = pd.DataFrame();
Outputs = pd.DataFrame();

Inputs = train_set.iloc[:,0:4];
Outputs = train_set.iloc[:,4::];

Inputs_test = test_set.iloc[:,0:4];
Outputs_test = test_set.iloc[:,4::];

model = Sequential();
model.add(Dense(8,input_shape=(4,), activation='relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(7,activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Inputs, Outputs, epochs=300, batch_size=10)

resultados = model.predict(Inputs_test)

erro_absoluto_nn = np.mean(100 * abs(resultados - Outputs_test)/Outputs_test)

_, accuracy = model.evaluate(Inputs, Outputs)
print('Accuracy: %.2f' % (accuracy*100))

