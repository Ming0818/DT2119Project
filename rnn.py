from network import Network
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import Sequential
from keras.layers import LSTM, Dense, Bidirectional


class RNN(Network):
    def train_model(self):
        es = EarlyStopping(patience=4)
        lr_reduce = ReduceLROnPlateau(factor=0.2, verbose=1)

        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(self.x_train.shape[1:])))
        model.add(LSTM(64))

        model.add(Dense(self.y_train.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        history_callback = model.fit(self.x_train, self.y_train,
                                     validation_data=(self.x_val, self.y_val),
                                     epochs=self.epochs,
                                     batch_size=2048, callbacks=[es, lr_reduce])
        self.train_loss_history = history_callback.history["loss"]
        self.train_acc_history = history_callback.history["acc"]
        return model
