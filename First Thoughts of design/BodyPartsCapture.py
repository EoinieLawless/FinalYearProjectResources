import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            lmPose = mp_pose.PoseLandmark

            # Define connections between landmarks
            connections = [
                (lmPose.LEFT_SHOULDER, lmPose.LEFT_ELBOW),
                (lmPose.LEFT_ELBOW, lmPose.LEFT_WRIST),
                (lmPose.RIGHT_SHOULDER, lmPose.RIGHT_ELBOW),
                (lmPose.RIGHT_ELBOW, lmPose.RIGHT_WRIST),
                (lmPose.LEFT_HIP, lmPose.LEFT_KNEE),
                (lmPose.LEFT_KNEE, lmPose.LEFT_ANKLE),
                (lmPose.RIGHT_HIP, lmPose.RIGHT_KNEE),
                (lmPose.RIGHT_KNEE, lmPose.RIGHT_ANKLE),
                (lmPose.LEFT_SHOULDER, lmPose.RIGHT_SHOULDER),
                (lmPose.LEFT_HIP, lmPose.RIGHT_HIP),
                # Add more connections as needed
            ]

            # Draw lines for each connection
            for connection in connections:
                start = connection[0]
                end = connection[1]
                x1, y1 = int(lm[start].x * frame.shape[1]), int(lm[start].y * frame.shape[0])
                x2, y2 = int(lm[end].x * frame.shape[1]), int(lm[end].y * frame.shape[0])
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Draw circles at each landmark
            for part in lmPose:
                x = int(lm[part].x * frame.shape[1])
                y = int(lm[part].y * frame.shape[0])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
