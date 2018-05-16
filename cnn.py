from network import Network
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import Sequential
import numpy as np
from keras.layers import Conv2D, Dropout, Flatten, Dense


class CNN(Network):

    def train_model(self):
        es = EarlyStopping(patience=4)
        lr_reduce = ReduceLROnPlateau(factor=0.2, verbose=1)
        self.x_train = np.expand_dims(self.x_train, 3)
        self.x_val = np.expand_dims(self.x_val, 3)
        self.x_test = np.expand_dims(self.x_test, 3)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                         activation='relu',
                         input_shape=(self.x_val.shape[1:])))
        model.add(Dropout(0.3))
        model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.y_val.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history_callback = model.fit(self.x_train, self.y_train,
                                     validation_data=(self.x_val, self.y_val),
                                     epochs=self.epochs,
                                     batch_size=2048, callbacks=[es, lr_reduce])
        return model
