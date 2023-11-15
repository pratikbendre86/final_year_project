import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

dataset = pd.read_csv('New_SET.csv')
X = dataset.iloc[:,0:4].values
Y = dataset.iloc[:,4].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
joblib.dump(sc, 'ann_project_scalar_NEW.gz')

classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Dense(units = 25, kernel_initializer= 'uniform', activation = 'relu', input_dim = 4))
classifier.add(tf.keras.layers.Dense(units = 25, kernel_initializer= 'uniform', activation = 'relu'))
classifier.add(tf.keras.layers.Dense(units = 5, kernel_initializer= 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X, Y, batch_size = 10, epochs = 200)
classifier.save("ANN_Project_NEW.h5")