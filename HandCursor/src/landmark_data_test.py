import cv2
import mediapipe as mp
import time

# Coordinate System Explanation:
# x: Horizontal position normalized between 0.0 (left edge) and 1.0 (right edge).
# y: Vertical position normalized between 0.0 (top edge) and 1.0 (bottom edge).
# z: Depth relative to the wrist (landmark 0). 
#    Negative z values mean the landmark is closer to the camera than the wrist.
#    Positive z values mean the landmark is further away from the camera.

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 21 Hand Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky and Palm
]

# Important landmarks as defined by MediaPipe
KEY_LANDMARKS = {
    0: "Wrist",
    4: "Thumb Tip",
    8: "Index Fingertip",
    12: "Middle Fingertip",
    16: "Ring Fingertip",
    20: "Pinky Fingertip"
}

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        timestamp_ms = 0
        last_print_time = time.time()

        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Could not read a frame.")
                break

            # Mirror the camera frame so it acts like a real mirror
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    height, width, _ = frame.shape
                    pixel_landmarks = []
                    
                    # 1. First, process the geometry to draw on the screen
                    for landmark in hand_landmarks:
                        cx = int(landmark.x * width)
                        cy = int(landmark.y * height)
                        pixel_landmarks.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                    
                    for connection in HAND_CONNECTIONS:
                        start_point = pixel_landmarks[connection[0]]
                        end_point = pixel_landmarks[connection[1]]
                        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

                    # 2. Second, print the coordinate data to terminal at intervals
                    current_time = time.time()
                    
                    # Only print if 500 milliseconds (0.5 seconds) have passed
                    if current_time - last_print_time >= 0.5:
                        print("\n" + "=" * 60)
                        print("Hand detected! Landmark coordinates:")
                        
                        # Loop through all 21 landmarks
                        for i, landmark in enumerate(hand_landmarks):
                            # Check if this index is in our KEY_LANDMARKS dictionary
                            name = KEY_LANDMARKS.get(i, "")
                            name_suffix = f" ({name})" if name else ""
                            
                            print(f"Index {i:2d}{name_suffix:18}: x={landmark.x:5.3f}, y={landmark.y:5.3f}, z={landmark.z:6.3f}")
                        
                        # Reset the timer
                        last_print_time = current_time

            cv2.imshow("HandCursor - Landmark Data Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
