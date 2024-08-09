import numpy as np
import tensorflow as tf

# Buat data x sebagai input
x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# Buat data y sebagai output (dengan noise)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Buat model sequential
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=500)

print(model.predict(np.array([10.0])))

evaluation = model.evaluate(x, y)
print("Evaluation result : ", evaluation)