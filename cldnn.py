from network import Network
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import Sequential
from keras.layers import LSTM, Dense, Conv2D, Reshape
import numpy as np


class CLDNN(Network):
    def train_model(self, conv_sizes, lstm_sizes, conv_kernels, dense_sizes):
        n_convs, n_lstm, n_dense = len(conv_sizes), len(lstm_sizes), len(dense_sizes)

        es = EarlyStopping(patience=4)
        lr_reduce = ReduceLROnPlateau(factor=0.2, verbose=1)

        self.x_train = np.expand_dims(self.x_train, 3)
        self.x_val = np.expand_dims(self.x_val, 3)
        self.x_test = np.expand_dims(self.x_test, 3)

        model = Sequential()  # conv, conv, dense, lstm, lstm, dnn, dnn, out
        model.add(Conv2D(conv_sizes[0], conv_kernels[0], input_shape=(self.x_val.shape[1:])))
        for i in range(1, n_convs):
            model.add(Conv2D(conv_sizes[i], conv_kernels[i]))

        # reshape after conv layers
        model.add(Reshape((self.context_length, -1)))

        model.add(LSTM(lstm_sizes[0], return_sequences=n_lstm > 1))  # if more than 1 layer, return sequences

        for i in range(1, n_lstm - 1):
            model.add(LSTM(lstm_sizes[i], return_sequences=True))
        if n_lstm > 1:
            model.add(LSTM(lstm_sizes[-1]))

        for i in range(n_dense):
            model.add(Dense(dense_sizes[i]))

        model.add(Dense(self.y_train.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(self.x_train, self.y_train,
                  validation_data=(self.x_val, self.y_val),
                  epochs=self.epochs,
                  batch_size=32, callbacks=[es, lr_reduce])
        return model
