# def train_lstm_cnn(self):
#     pass
#
# def train_cnn(self):
#     es = EarlyStopping(patience=4)
#     lr_reduce = LearningRateScheduler()
#     model = Sequential()
#     self.x_train = np.expand_dims(self.x_train, 3)
#     self.x_val = np.expand_dims(self.x_val, 3)
#     self.x_test = np.expand_dims(self.x_test, 3)
#     model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
#                      activation='relu',
#                      input_shape=(self.x_val.shape[1:])))
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
#     model.add(Dense(self.y_val.shape[1], activation='softmax'))
#
#     model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#
#     history_callback = model.fit(self.x_val, self.y_val,
#                                  validation_data=(self.x_val, self.y_val),
#                                  epochs=self.epochs,
#                                  batch_size=2048)