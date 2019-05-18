from keras.models import Sequential
from keras.layers import Activation, Dense
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='random_uniform'))
