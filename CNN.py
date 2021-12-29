import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten
from sklearn.preprocessing import StandardScaler

titanic = pd.read_csv("titanic.csv")

###TODO drop what needed
X = titanic.drop(['Survived'], axis=1)
Y = titanic.Survived


ss = StandardScaler()
X = ss.fit_transform(X)
y_train_onehot = pd.get_dummies(Y).values


from keras import optimizers
img_rows, img_cols = 2,2
nb_filters = 1000
pool_size = (1, 1)
kernel_size = (1, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('tanh'))
model.add(Dropout(0.25))
model.add(Dense(2))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['accuracy'])
model.fit(X_train, y_train_onehot, epochs=30)

model.fit(X_train, y_train_onehot, epochs=15, validation_split=0.1, batch_size=20)