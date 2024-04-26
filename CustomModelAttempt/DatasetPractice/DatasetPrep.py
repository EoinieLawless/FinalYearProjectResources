import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

num_classes = 10  # Total classes
num_keypoints = 8  # Total keypoints per image
image_size = (64, 64)  # Image size to resize to

# Placeholder function for loading keypoints
def load_keypoints(image_filename):
    # You need to implement this function
    # It should return keypoints as an array of shape (num_keypoints * 2,)
    return np.random.rand(num_keypoints * 2)  # Dummy data for demonstration

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size=32, shuffle=True, image_size=(64, 64)):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_size = image_size
        self.image_filenames = []
        self.classes = []
        self.keypoints = []
        
        # Load dataset
        self._load_dataset()
        
        self.on_epoch_end()
    
    def _load_dataset(self):
        # Assuming directory structure is: directory/class_name/image.jpg
        for class_idx, class_name in enumerate(sorted(os.listdir(self.directory))):
            class_dir = os.path.join(self.directory, class_name)
            if os.path.isdir(class_dir):
                for img_filename in os.listdir(class_dir):
                    self.image_filenames.append(os.path.join(class_dir, img_filename))
                    self.classes.append(class_idx)
                    # Load keypoints for each image here
                    self.keypoints.append(load_keypoints(img_filename))
        
        self.classes = tf.keras.utils.to_categorical(self.classes, num_classes=num_classes)
        self.keypoints = np.array(self.keypoints)
    
    def __len__(self):
        return len(self.image_filenames) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.image_filenames))
            np.random.shuffle(indices)
            self.image_filenames = [self.image_filenames[idx] for idx in indices]
            self.classes = self.classes[indices]
            self.keypoints = self.keypoints[indices]
    
    def __getitem__(self, index):
        batch_image_filenames = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = []
        for filename in batch_image_filenames:
            img = tf.keras.preprocessing.image.load_img(filename, target_size=self.image_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img /= 255.0
            batch_images.append(img)
        
        batch_classes = self.classes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_keypoints = self.keypoints[index * self.batch_size:(index + 1) * self.batch_size]
        
        print("Batch images shape:", np.array(batch_images).shape)
        print("Batch classes shape:", batch_classes.shape)
        print("Batch keypoints shape:", batch_keypoints.shape)
        
        return np.array(batch_images), {'class_output': batch_classes, 'keypoints_output': batch_keypoints}
    
    

# Paths to your dataset directories
train_dir = './TrainingSet'
validation_dir = './ValidationSet'

# Create data generators
train_generator = CustomDataGenerator(train_dir, batch_size=16, shuffle=True, image_size=image_size)
validation_generator = CustomDataGenerator(validation_dir, batch_size=16, shuffle=False, image_size=image_size)

def create_model(input_shape, num_classes, num_keypoints):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                    include_top=False,
                                                    weights='imagenet')
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)
    keypoints_output = layers.Dense(num_keypoints * 2, name='keypoints_output')(x)
    
    model = models.Model(inputs=inputs, outputs=[class_output, keypoints_output])
    
    return model

model = create_model((64, 64, 3), num_classes, num_keypoints)
model.compile(optimizer='adam',
              loss={'class_output': 'categorical_crossentropy',  # Adjusted from sparse_categorical_crossentropy
                    'keypoints_output': 'mse'},
              metrics={'class_output': 'accuracy',
                       'keypoints_output': 'mse'})


                       
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    epochs=10,
                    batch_size=16)


model.save('my_NewModel.h5')
