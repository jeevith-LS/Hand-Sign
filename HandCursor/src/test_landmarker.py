import mediapipe as mp

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    num_hands=1,
)

print("Creating Hand Landmarker...")

detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

print("Landmarker initialized successfully!")

detector.close()

print("Landmarker closed successfully!")