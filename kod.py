import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import dagshub
import mlflow.tensorflow


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

dagshub.init(repo_owner='OlivierArthur', repo_name='CNN_cifar10_porownanie', mlflow=True)
mlflow.set_experiment("CNN_CIFAR10_porownanie")
mlflow.tensorflow.autolog()

experiments = [
    {
        "name": "Eksperyment_1_prosty",
        "layers": [
            layers.Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ]
    },
    {
        "name": "Eksperyment_2_glebszy",
        "layers": [
            layers.Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10)
        ]
    },
    {
        "name": "Eksperyment_3_Dropout",
        "layers": [
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ]
    },
    {
        "name": "Eksperyment_4_Augmentacja",
        "layers": [
            #augmentacja
            layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
            layers.RandomRotation(0.1),


            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(10, activation='softmax')
        ]
    },
]

for exp in experiments:
  with mlflow.start_run(run_name=exp['name']):
    model = models.Sequential(exp['layers'])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=20,
              validation_data=(test_images, test_labels), verbose=1)


