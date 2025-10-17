import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.sparse import load_npz
import joblib
import os

X = load_npz('artifacts/X_combined.npz')
y = np.load('artifacts/y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_dim = X_train.shape[1]

model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

def make_dataset(X, y, batch_size=64, shuffle=True):
    X = X.tocsr()
    def gen():
        for i in range(X.shape[0]):
            yield X.getrow(i).toarray().reshape(-1).astype('float32'), y[i].astype('float32')
    ds = tf.data.Dataset.from_generator(gen,
        output_signature=(tf.TensorSpec(shape=(X.shape[1],), dtype=tf.float32),
                          tf.TensorSpec(shape=(), dtype=tf.float32)))
    if shuffle:
        ds = ds.shuffle(2048)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_train, y_train)
val_ds = make_dataset(X_test, y_test, shuffle=False)

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

os.makedirs('artifacts', exist_ok=True)
model.save('artifacts/deepcsat_model.keras')
print("✅ Model saved to artifacts/deepcsat_model.keras")


preds = []
for i in range(X_test.shape[0]):
    preds.append(model.predict(X_test.getrow(i).toarray().astype('float32'), verbose=0)[0, 0])
preds = np.array(preds)

print("MAE:", mean_absolute_error(y_test, preds))
print("RMSE:", mean_squared_error(y_test, preds, squared=False))
print("R²:", r2_score(y_test, preds))
