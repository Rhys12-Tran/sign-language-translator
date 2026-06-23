"""
hand_detector.py
-----------------
Phase 1 of the ASL Recognizer project.

What this script does:
  1. Turns on the webcam.
  2. Asks MediaPipe to find a hand in every video frame and locate its
     21 "landmarks" (fingertip, knuckle, wrist, etc. positions).
  3. Draws the 21 landmark dots and the lines ("bones") connecting them
     on top of the live video.
  4. Prints the wrist landmark's x/y position to the terminal every frame.
  5. Quits cleanly when you press "q".

Note on the MediaPipe version installed in this project: it ships hand
detection through the newer "Tasks" API (mp.tasks.vision.HandLandmarker)
instead of the older "mp.solutions.hands" API you may see in older
tutorials online. The newer API needs a small pre-trained model file,
which has already been downloaded to this folder as "hand_landmarker.task".
"""

import os                # Used to build a reliable path to the model file
import time              # Used to create increasing timestamps MediaPipe needs for video mode

import cv2               # OpenCV: gives us webcam access and lets us draw on / display frames
import mediapipe as mp   # MediaPipe: Google's library that detects hands and their landmarks

# ---------------------------------------------------------------------------
# STEP 1: Build the hand landmark detector
# ---------------------------------------------------------------------------

# Pull out the specific classes we need so the rest of the file reads cleanly.
BaseOptions = mp.tasks.BaseOptions                             # Shared settings for all MediaPipe tasks
HandLandmarker = mp.tasks.vision.HandLandmarker                 # The hand-detector class itself
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions   # Settings specific to hand detection
VisionRunningMode = mp.tasks.vision.RunningMode                 # Tells MediaPipe "this is live video", not one photo

# These two know how to turn a list of landmark coordinates into the
# dots-and-lines drawing you see in hand-tracking demos.
drawing_utils = mp.tasks.vision.drawing_utils
drawing_styles = mp.tasks.vision.drawing_styles

# Which landmark dots should be connected by a line (e.g. fingertip to the
# knuckle below it). 0 is always the wrist.
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

# Build a full path to the model file based on where THIS script lives, so
# the script works no matter which folder you run it from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

# Configure the detector: where the model file is, that we're feeding it a
# live video stream (so it can track smoothly between frames), and that we
# only want to track 1 hand at a time (simpler + faster for a one-hand
# alphabet recognizer).
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)

# Actually create the detector using the options above.
landmarker = HandLandmarker.create_from_options(options)

# ---------------------------------------------------------------------------
# STEP 2: Open the webcam
# ---------------------------------------------------------------------------

cap = cv2.VideoCapture(0)   # "0" = the first/default camera plugged into this computer

if not cap.isOpened():
    # This usually means another app is using the webcam, or there is no
    # camera at index 0 (try changing the 0 above to 1 if you have more
    # than one camera).
    print("ERROR: could not open the webcam. Is it being used by another app?")
    raise SystemExit(1)

# MediaPipe's VIDEO mode needs a timestamp (in milliseconds) for every frame,
# and that timestamp must keep increasing. We measure elapsed time since the
# webcam started so each frame gets a bigger number than the last.
start_time = time.time()

print("Webcam started. Press 'q' in the video window to quit.")

# ---------------------------------------------------------------------------
# STEP 3: Main loop -- read a frame, detect the hand, draw on it, show it
# ---------------------------------------------------------------------------

while True:
    success, frame = cap.read()   # Grab one frame from the webcam as a BGR image
    if not success:
        print("ERROR: failed to read a frame from the webcam.")
        break

    # OpenCV stores color as BGR, but MediaPipe expects RGB, so swap the
    # color channels before handing the frame to the detector.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap the raw pixel array in MediaPipe's own Image container.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # How many milliseconds have passed since the webcam started.
    timestamp_ms = int((time.time() - start_time) * 1000)

    # Run hand detection on this frame.
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # result.hand_landmarks is a list of hands found in the frame. Since we
    # set num_hands=1, it will contain either 0 hands or 1 hand.
    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]   # The 21 landmarks of the one detected hand

        # Draw the 21 dots and connecting lines directly onto the displayed frame.
        drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )

        # Landmark index 0 is always the wrist. x and y are "normalized",
        # meaning they range from 0.0 to 1.0 no matter the camera resolution
        # (0.5, 0.5 is the exact center of the frame).
        wrist = hand_landmarks[0]
        print(f"Wrist position -> x: {wrist.x:.3f}, y: {wrist.y:.3f}")
    else:
        print("No hand detected.")

    # Show the frame (now with landmark drawings) in a window.
    cv2.imshow("ASL Hand Detector", frame)

    # Wait 1 millisecond for a key press. If the key was "q", stop the loop.
    # "& 0xFF" keeps only the lowest 8 bits, which is the normal way to
    # compare OpenCV key codes across different operating systems.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------------------------------------------------------------------
# STEP 4: Clean up so the webcam and windows are released properly
# ---------------------------------------------------------------------------

cap.release()              # Free the webcam so other programs can use it
cv2.destroyAllWindows()    # Close the video window
landmarker.close()         # Free MediaPipe's internal resources

print("Webcam closed. Goodbye!")
