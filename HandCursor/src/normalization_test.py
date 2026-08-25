import cv2
import mediapipe as mp
import time
import math

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

# Mathematical Normalization Explanation:
# Problem: As your hand moves closer to the camera, it appears larger in the video frame.
# This causes the raw pixel/normalized distance between your thumb and index finger to increase,
# even if your fingers haven't actually moved further apart.
#
# Solution (Normalization): 
# We need a "reference distance" on the hand that scales exactly the same way.
# The distance from the wrist (0) to the middle fingertip (12) is a great reference for overall hand size.
# 
# 1. thumb_index_distance = Distance between thumb (4) and index (8).
# 2. hand_size = Distance between wrist (0) and middle fingertip (12).
# 3. normalized_distance = thumb_index_distance / hand_size
#
# Because both distances scale together as the hand moves toward or away from the camera,
# dividing them cancels out the depth factor. The normalized_distance will remain 
# constant regardless of how close your hand is to the webcam!

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
                    
                    # 1. Calculate and Draw all 21 points
                    for landmark in hand_landmarks:
                        cx = int(landmark.x * width)
                        cy = int(landmark.y * height)
                        pixel_landmarks.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                    
                    # 2. Draw standard connections
                    for connection in HAND_CONNECTIONS:
                        start_point = pixel_landmarks[connection[0]]
                        end_point = pixel_landmarks[connection[1]]
                        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                        
                    # 3. Extract the 4 specific landmarks we care about
                    wrist = hand_landmarks[0]
                    thumb_tip = hand_landmarks[4]
                    index_tip = hand_landmarks[8]
                    middle_tip = hand_landmarks[12]
                    
                    # 4. Perform the mathematical distances
                    thumb_index_distance = calculate_distance(thumb_tip, index_tip)
                    hand_size = calculate_distance(wrist, middle_tip)
                    
                    # Prevent division by zero if hand_size is somehow 0
                    normalized_distance = (thumb_index_distance / hand_size) if hand_size > 0 else 0.0
                    
                    # 5. Draw highlighted lines for the two distances we measured
                    cv2.line(frame, pixel_landmarks[4], pixel_landmarks[8], (0, 255, 255), 3) # Yellow for thumb-index
                    cv2.line(frame, pixel_landmarks[0], pixel_landmarks[12], (255, 0, 255), 3) # Magenta for hand size
                    
                    # 6. Display metrics on the video frame
                    cv2.putText(frame, f"Thumb-Index Dist: {thumb_index_distance:.3f}", (30, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Hand Size: {hand_size:.3f}", (30, 75), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(frame, f"Normalized: {normalized_distance:.3f}", (30, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 7. Terminal logging every 500ms
                    current_time = time.time()
                    if current_time - last_print_time >= 0.5:
                        print("-" * 40)
                        print(f"Thumb-Index Dist: {thumb_index_distance:.4f}")
                        print(f"Hand Size:        {hand_size:.4f}")
                        print(f"Normalized:       {normalized_distance:.4f}")
                        last_print_time = current_time

            cv2.imshow("HandCursor - Normalization Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
