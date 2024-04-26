import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your pre-trained model
model_path = './my_NewModel.h5'
model = load_model(model_path)

# Class names, replace these with your actual class names
class_names = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6','Class7','Class8',]

# Path to your image file
image_path = './body.jpg'

# Load the image
frame = cv2.imread(image_path)

if frame is not None:
    # Preprocess the frame for the model
    input_frame = cv2.resize(frame, (64, 64))
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)

    # Predict with the model
    predictions = model.predict(input_frame)
    class_predictions, keypoints_predictions = predictions

    # Determine the predicted class
    predicted_class_index = np.argmax(class_predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    keypoints = keypoints_predictions[0]

    # Display the class name on the image
    cv2.putText(frame, f'Class: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if keypoints.shape[0] == 8:  # Assuming 8 keypoints
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i] * frame.shape[1])
            y = int(keypoints[i + 1] * frame.shape[0])
            cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

        # Also display how many keypoints were detected
        cv2.putText(frame, f'Keypoints: {len(keypoints)//2}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Keypoints and Class', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("The model's output has an unexpected shape:", keypoints.shape)
else:
    print("Failed to load the image")
