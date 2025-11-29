import tensorflow as tf

def build_lstm(window: int, n_features: int):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            64,
            return_sequences=True,
            input_shape=(window, n_features)
        ),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model


def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
    ]