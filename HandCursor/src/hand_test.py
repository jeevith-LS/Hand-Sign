import cv2
import mediapipe as mp

# MediaPipe Tasks API requires specific configuration objects
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Define the standard 21 hand landmark connections manually
# Each tuple represents a line connecting two landmark indices
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky and palm base
]

# Configure the Hand Landmarker for processing a video stream
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open the Mac camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

try:
    # Use a 'with' block to automatically clean up the landmarker resources
    with HandLandmarker.create_from_options(options) as landmarker:
        timestamp_ms = 0

        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Could not read a frame.")
                break

            # Mirror the camera image
            frame = cv2.flip(frame, 1)

            # Convert OpenCV BGR format to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # The Tasks API requires a specific 'mp.Image' object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # The Tasks API in VIDEO mode requires an increasing timestamp in milliseconds
            # We increment by 33ms (roughly 30 FPS) for each frame
            timestamp_ms += 33

            # Process the image to detect hands
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw the landmarks manually if any hands were detected
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    height, width, _ = frame.shape
                    
                    # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates
                    pixel_landmarks = []
                    for landmark in hand_landmarks:
                        cx = int(landmark.x * width)
                        cy = int(landmark.y * height)
                        pixel_landmarks.append((cx, cy))
                        
                        # Draw the landmark point (a small filled green circle)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                    
                    # Draw the connections (blue lines between points)
                    for connection in HAND_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        
                        start_point = pixel_landmarks[start_idx]
                        end_point = pixel_landmarks[end_idx]
                        
                        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

            # Display the camera frame
            cv2.imshow("HandCursor - Tasks API Test", frame)

            # Press Q to quit safely
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
