import tensorflow as tf
from tensorflow import keras

from keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory

num_keypoints = 8

train_dir = '.\TrainingSet'
validation_dir = '.\TrainingSet'

# Loading and preprocessing the training data
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=16,
    image_size=(64, 64))  

# Loading and preprocessing the validation data
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    shuffle=True,
    batch_size=16,
    image_size=(64, 64))  

# Assuming you have a way to load keypoints data, you'll need to merge it with the image dataset
y_train_keypoints = np.random.rand(100, num_keypoints * 2) 
y_val_keypoints = np.random.rand(20, num_keypoints * 2) 


# Assuming you have preloaded and preprocessed your dataset
# Images of shape 64x64 for example, with 3 channels (RGB)
# Adjust the shapes and sizes according to your actual dataset
num_classes = 8  # e.g., 5 body parts you want to classify
num_keypoints = 8  # e.g., 4 keypoints per image you want to detect


X_train = np.random.rand(100, 64, 64, 3)  # 100 training images
y_train_class = np.random.randint(0, num_classes, 100)  # Class labels
y_train_keypoints = np.random.rand(100, num_keypoints * 2)  # Keypoints

X_val = np.random.rand(20, 64, 64, 3)  # 20 validation images
y_val_class = np.random.randint(0, num_classes, 20)  # Validation class labels
y_val_keypoints = np.random.rand(20, num_keypoints * 2)  # Validation keypoints

# Define a CNN model
def create_model(input_shape, num_classes, num_keypoints):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                    include_top=False,
                                                    weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Branch for classification
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)
    
    # Branch for keypoints detection
    keypoints_output = layers.Dense(num_keypoints * 2, name='keypoints_output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=[class_output, keypoints_output])
    
    return model

model = create_model((64, 64, 3), num_classes, num_keypoints)
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss={'class_output': 'sparse_categorical_crossentropy',
                    'keypoints_output': 'mse'},
              metrics={'class_output': 'accuracy',
                       'keypoints_output': 'mse'})

# Train the model
history = model.fit(X_train, {'class_output': y_train_class, 'keypoints_output': y_train_keypoints},
                    validation_data=(X_val, {'class_output': y_val_class, 'keypoints_output': y_val_keypoints}),
                    epochs=10,
                    batch_size=16)


model.save('my_model.h5')

