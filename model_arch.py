import tensorflow as tf
from tensorflow.keras import layers

resizer = lambda x: tf.image.resize_with_pad((x / 255.), 224, 224)

resize_and_rescale = tf.keras.Sequential([
  tf.keras.layers.Lambda(resizer)
])

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.GaussianNoise(0.35),
  tf.keras.layers.experimental.preprocessing.RandomRotation((-0.5,0.5)),
  tf.keras.layers.experimental.preprocessing.RandomTranslation((-0.1,0.1), (-0.1,0.1))
])


baseline_model = tf.keras.models.Sequential([
  layers.Conv2D(16, 12, strides=3, padding='same', activation='relu'),
  layers.Conv2D(32, 7, strides=2, padding='same', activation='relu'),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dropout(0.3),
  layers.Dense(256, activation='relu'),
  layers.Dropout(0.3),
  layers.Dense(4, activation="softmax")
])


model = tf.keras.Sequential([
      resize_and_rescale,
      data_augmentation,
      baseline_model])
