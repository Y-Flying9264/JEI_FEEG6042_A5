"""
Model definitions for MNIST experiments [Question 5]:
1) Sequential fully connected network (Dense baseline)
2) Sequential convolutional neural network (CNN)
3) Parallel fully connected network with two branches
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_dense_model():
    """
    Sequential fully connected baseline:
    Input(784) -> Dense(256, relu) -> Dense(64, relu) -> Dense(10, softmax)
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(28 * 28,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_model():
    """
    Sequential convolutional model:
    Input(28, 28, 1) ->
      Conv2D(16) -> Conv2D(32) -> MaxPool(2x2) ->
      Flatten -> Dense(32, relu) -> Dense(10, softmax)
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPool2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_parallel_dense_model():
    """
    Parallel fully connected model with two branches:
    Input(784) -> [Dense(128), Dense(128)] in parallel ->
    Concatenate(256) -> Dense(10, softmax)
    """
    inputs = keras.Input(shape=(28 * 28,))

    branch_1 = layers.Dense(128, activation="relu")(inputs)
    branch_2 = layers.Dense(128, activation="relu")(inputs)

    merged = layers.Concatenate()([branch_1, branch_2])
    outputs = layers.Dense(10, activation="softmax")(merged)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Quick sanity check
    dense = build_dense_model()
    cnn = build_cnn_model()
    parallel = build_parallel_dense_model()

    print("=== Dense model summary ===")
    dense.summary()
    print("\n=== CNN model summary ===")
    cnn.summary()
    print("\n=== Parallel Dense model summary ===")
    parallel.summary()
