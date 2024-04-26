import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
print("Current Working Directory:", os.getcwd())


# Load your pre-trained model
model_path = './my_model.h5'
model = load_model(model_path)

# Path to your image file
image_path = './body.jpg'  # Update this to the path of your photo

# Load the image
frame = cv2.imread(image_path)

if frame is not None:
    # Preprocess the frame for the model
    input_frame = cv2.resize(frame, (64, 64))  # Resize to match the model's expected input
    input_frame = input_frame / 255.0  # Normalize if your model expects values in [0, 1]
    input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension

    # Use the model to predict the class of the image
    prediction = model.predict(input_frame)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Assuming your model outputs class indices
    
    print("Prediction:", prediction)
    print("Predicted Class:", predicted_class)

    # Example: Display the predicted class on the image
    cv2.putText(frame, f"Predicted Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the image
    cv2.imshow('Predicted Image', frame)
    
    # Wait for a key press and close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load the image")
