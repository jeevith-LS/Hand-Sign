import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read a frame.")
            break

        # Mirror the camera image
        frame = cv2.flip(frame, 1)

        # Display the camera frame
        cv2.imshow("HandCursor - Camera Test", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Always release the camera and close OpenCV windows
    camera.release()
    cv2.destroyAllWindows()