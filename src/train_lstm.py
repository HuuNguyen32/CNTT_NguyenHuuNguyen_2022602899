import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)  

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATA_DIR = "data/processed_npy"

LABELS = {
    "hand_raising": 0,
    "reading": 1,
    "writing": 2,
    "sleeping": 3
}

X = []
y = []

for label in LABELS:

    folder = os.path.join(DATA_DIR, label)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        data = np.load(path)

        X.append(data)

        y.append(LABELS[label])

X = np.array(X)

y = to_categorical(y)

model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(30, 46)))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(Dense(4, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X, y,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

model.save("models/lstm/student_behavior_lstm.h5")
