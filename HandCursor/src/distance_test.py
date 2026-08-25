import cv2
import mediapipe as mp
import time
import math

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# Mathematical Distance Explanation:
# To find the straight-line distance between two points (x1, y1) and (x2, y2), 
# we use the Euclidean distance formula based on the Pythagorean theorem (a^2 + b^2 = c^2).
# 
# 1. Calculate the horizontal difference: dx = x2 - x1
# 2. Calculate the vertical difference: dy = y2 - y1
# 3. Square both differences: dx^2 + dy^2
# 4. Take the square root of the sum: distance = sqrt(dx^2 + dy^2)
#
# Because we are using normalized coordinates (values between 0.0 and 1.0),
# the resulting distance will typically be a small decimal (e.g., 0.02 when pinching, 
# or 0.30 when fingers are spread apart).
def calculate_distance(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.sqrt(dx**2 + dy**2)

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

            # Mirror the camera frame
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    height, width, _ = frame.shape
                    pixel_landmarks = []
                    
                    # Draw all 21 landmarks
                    for landmark in hand_landmarks:
                        cx = int(landmark.x * width)
                        cy = int(landmark.y * height)
                        pixel_landmarks.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                    
                    # Draw normal hand connections
                    for connection in HAND_CONNECTIONS:
                        start_point = pixel_landmarks[connection[0]]
                        end_point = pixel_landmarks[connection[1]]
                        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                        
                    # Extract Landmark 4 (Thumb Tip) and Landmark 8 (Index Fingertip)
                    thumb_tip = hand_landmarks[4]
                    index_tip = hand_landmarks[8]
                    
                    # Calculate their 2D Euclidean distance using normalized x/y coordinates
                    distance = calculate_distance(thumb_tip, index_tip)
                    
                    # Draw a distinct yellow line directly between the thumb and index finger
                    thumb_pixel = pixel_landmarks[4]
                    index_pixel = pixel_landmarks[8]
                    cv2.line(frame, thumb_pixel, index_pixel, (0, 255, 255), 3)
                    
                    # Display the distance text on the camera frame
                    cv2.putText(frame, f"Distance: {distance:.3f}", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Print the distance to the terminal approx. every 500 ms
                    current_time = time.time()
                    if current_time - last_print_time >= 0.5:
                        print(f"Thumb to Index Distance: {distance:.4f}")
                        last_print_time = current_time

            cv2.imshow("HandCursor - Distance Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
