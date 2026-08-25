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

# === Gesture Recognition Concepts ===
# 
# 1. Why normalized distances are used:
#    If we used raw distances, a pinch right in front of the camera would have a larger distance 
#    than an open hand far away from the camera. Normalizing by the overall hand size (wrist to 
#    middle fingertip) cancels out the depth factor, making the gesture work reliably at any distance.
#
# 2. Why two thresholds are used & What hysteresis means:
#    Hysteresis creates a "dead zone" using two thresholds (PINCH_THRESHOLD and RELEASE_THRESHOLD).
#    You must squeeze tight (< 0.20) to trigger a pinch, but you must open wide (> 0.25) to release it.
#    This prevents the gesture state from rapidly flickering ("chattering") if your hand jitters 
#    right on the edge of a single threshold.
#
# 3. Why consecutive-frame confirmation is needed:
#    The webcam or the machine learning model might output a single glitchy frame where the fingers 
#    appear pinched. Requiring the condition to remain stable for several consecutive frames 
#    (temporal smoothing) filters out these micro-glitches and prevents accidental phantom clicks.
#
# 4. Why RIGHT_PINCH must have priority over LEFT_PINCH:
#    A 3-finger pinch (RIGHT_PINCH) inherently includes a 2-finger pinch (thumb + index). 
#    If we checked for the 2-finger pinch first, the code would see the thumb and index touching, 
#    immediately classify it as a LEFT_PINCH, and ignore the middle finger entirely. By checking the 
#    harder, more complex condition first, we correctly identify when all three fingers are touching.

PINCH_THRESHOLD = 0.20
RELEASE_THRESHOLD = 0.25
REQUIRED_CONSECUTIVE_FRAMES = 4

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
        
        # Advanced State Tracking
        current_state = "OPEN"
        potential_next_state = "OPEN"
        consecutive_frames = 0

        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Could not read a frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            norm_thumb_index = 0.0
            norm_thumb_middle = 0.0

            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    height, width, _ = frame.shape
                    pixel_landmarks = []
                    
                    for landmark in hand_landmarks:
                        cx = int(landmark.x * width)
                        cy = int(landmark.y * height)
                        pixel_landmarks.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                    
                    for connection in HAND_CONNECTIONS:
                        start_point = pixel_landmarks[connection[0]]
                        end_point = pixel_landmarks[connection[1]]
                        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                        
                    wrist = hand_landmarks[0]
                    thumb_tip = hand_landmarks[4]
                    index_tip = hand_landmarks[8]
                    middle_tip = hand_landmarks[12]
                    
                    dist_thumb_index = calculate_distance(thumb_tip, index_tip)
                    dist_thumb_middle = calculate_distance(thumb_tip, middle_tip)
                    hand_size = calculate_distance(wrist, middle_tip)
                    
                    if hand_size > 0:
                        norm_thumb_index = dist_thumb_index / hand_size
                        norm_thumb_middle = dist_thumb_middle / hand_size
                    
                    # Highlight active gesture lines
                    cv2.line(frame, pixel_landmarks[4], pixel_landmarks[8], (0, 255, 255), 3) # Yellow: Thumb-Index
                    cv2.line(frame, pixel_landmarks[4], pixel_landmarks[12], (255, 0, 255), 3) # Magenta: Thumb-Middle

                    # === Advanced Multi-State Hysteresis Logic ===
                    # Start by assuming the state won't change
                    intended_state = current_state

                    # Priority 1: Right Pinch (3 fingers touching)
                    if norm_thumb_index < PINCH_THRESHOLD and norm_thumb_middle < PINCH_THRESHOLD:
                        intended_state = "RIGHT_PINCH"
                    
                    # Priority 2: Left Pinch (2 fingers touching, but middle finger safely released)
                    elif norm_thumb_index < PINCH_THRESHOLD and norm_thumb_middle > RELEASE_THRESHOLD:
                        intended_state = "LEFT_PINCH"
                    
                    # Priority 3: Open (Both fingers safely released)
                    elif norm_thumb_index > RELEASE_THRESHOLD and norm_thumb_middle > RELEASE_THRESHOLD:
                        intended_state = "OPEN"
                    
                    # NOTE: If we are in the dead zone (e.g. distance is 0.22), intended_state remains current_state.

                    # === Temporal Smoothing (Consecutive Frames) ===
                    if intended_state == potential_next_state:
                        consecutive_frames += 1
                    else:
                        potential_next_state = intended_state
                        consecutive_frames = 1

                    # Apply state transition if the condition has been held long enough
                    if consecutive_frames >= REQUIRED_CONSECUTIVE_FRAMES and current_state != potential_next_state:
                        print(f"Gesture changed: {current_state} -> {potential_next_state}")
                        current_state = potential_next_state
                    
            # Display metrics on screen based on state
            if current_state == "RIGHT_PINCH":
                color = (255, 0, 255) # Magenta
            elif current_state == "LEFT_PINCH":
                color = (0, 255, 255) # Yellow
            else:
                color = (0, 255, 0) # Green
                
            cv2.putText(frame, f"State: {current_state}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(frame, f"Thumb-Index:  {norm_thumb_index:.3f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Thumb-Middle: {norm_thumb_middle:.3f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            cv2.imshow("HandCursor - Advanced Pinch Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
