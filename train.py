import numpy as np
import keras
from keras import layers
from keras import optimizers

def load():
    data = np.load("processed/dataset_2M.npz")
    X = data['arr_0']
    Y = data['arr_1']

    X_train, Y_train = X[:-10000], Y[:-10000]
    X_test, Y_test = X[-10000:], Y[-10000:]

    return X_train, Y_train, X_test, Y_test

def build_nn():
    nn = keras.Sequential()

    nn.add(layers.Input(shape=(5, 8, 8)))

    # 16
    nn.add(
        layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding='same',
            activation='relu',
        )
    )
    nn.add(
        layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding='same',
            activation='relu',
        )
    )

    # 32
    nn.add(
        layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding='same',
            activation='relu',
        )
    )
    nn.add(
        layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding='same',
            activation='relu',
        )
    )
    nn.add(
        layers.Conv2D(
            filters=32,
            kernel_size=3,
            activation='relu',
        )
    )

    # 64
    nn.add(
        layers.Conv2D(
            filters=64,
            kernel_size=2,
            padding='same',
            activation='relu',
        )
    )
    nn.add(
        layers.Conv2D(
            filters=64,
            kernel_size=2,
            padding='same',
            activation='relu',
        )
    )
    nn.add(
        layers.Conv2D(
            filters=64,
            kernel_size=2,
            activation='relu',
        )
    )

    # 128
    nn.add(
        layers.Conv2D(
            filters=128,
            kernel_size=1,
            activation='relu',
        )
    )
    nn.add(
        layers.Conv2D(
            filters=128,
            kernel_size=1,
            activation='relu',
        )
    )
    nn.add(
        layers.Conv2D(
            filters=128,
            kernel_size=1,
            activation='relu',
        )
    )

    nn.add(layers.Flatten())

    nn.add(layers.Dropout(0.3))
    nn.add(layers.Dense(units=1024, activation='relu'))
    nn.add(layers.Dropout(0.3))
    nn.add(layers.Dense(units=512, activation='relu'))

    nn.add(layers.Dense(units=1, activation='tanh'))

    nn.compile(optimizer=optimizers.Adam(lr=0.001), loss='mean_squared_error', metrics=['accuracy'])

    return nn

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load()

    nn = build_nn()

    nn.fit(
        x=X_train,
        y=Y_train,
        validation_data=(X_test, Y_test),
        epochs=100,
        batch_size=256,
        shuffle=True
    )