from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from ExtractMFCC import getMFCCFeature
from OneHotEncoding import getOneHotLabel
import matplotlib.pyplot as plt

class LSTMModel:
    def __int__(self):
        self.model = Sequential([
            LSTM(123, input_shape=(40, 1), return_sequences=False),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(7, activation='softmax')
        ])

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # %%
        self.history = self.model.fit(
            getMFCCFeature(),
            getOneHotLabel(),
            epochs=100,
            batch_size=256,
            shuffle=True
        )
        def showLossAndAccuracy(self):
            epochs = list(range(100))
            acc = self.history.history['accuracy']
            val_acc = self.history.history['val_accuracy']

            plt.plot(epochs, acc, label='train accuracy')
            plt.plot(epochs, val_acc, label='val accuracy')
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.legend()
            plt.show()
            return epochs, acc, val_acc

        def showLoss(self):
            epochs = list(range(100))
            loss = self.history.history['loss']
            val_loss = self.history.history['val_loss']

            plt.plot(epochs, loss, label='train loss')
            plt.plot(epochs, val_loss, label='val loss')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.show()
