import tensorflow as tf
from tensorflow import keras

print('tensorflow version:',tf.__version__)
##from tensorflow.python.compiler.mlcompute import mlcompute

tf.config.run_functions_eagerly(False)

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(2048,activation='relu'),
        keras.layers.Dense(4096,activation='relu'),
        keras.layers.Dense(2048,activation='relu'),
        keras.layers.Dense(10,activation='softmax')
    ]
)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              )

model.summary()

#%%time
model.fit(x_train,y_train,epochs=5,batch_size=1024)



