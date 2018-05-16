from network import Network
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import Sequential
import numpy as np
from keras.layers import Conv2D, Flatten, Dense


class CNN(Network):

    def train_cnn(self):
        es = EarlyStopping(patience=4)
        lr_reduce = ReduceLROnPlateau(factor=0.2, verbose=1)
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=(self.x_val.shape[1:])))
        model.add(Flatten())
        model.add(Dense(self.y_val.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        history_callback = model.fit(self.x_val, self.y_val,
                                     validation_data=(self.x_val, self.y_val),
                                     epochs=self.epochs,
                                     batch_size=2048, callbacks=[es, lr_reduce])
        return model