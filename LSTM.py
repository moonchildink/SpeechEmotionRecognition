from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from ExtractMFCC import getMFCCFeature
from OneHotEncoding import getOneHotLabel


model = Sequential([
    LSTM(123,input_shape=(40,1),return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
print(model.summary())

# %%
history = model.fit(
    getMFCCFeature(),
    getOneHotLabel(),
    epochs=100,
    batch_size=256,
    shuffle=True
)