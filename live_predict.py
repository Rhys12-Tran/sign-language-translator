"""
live_predict.py
-----------------
Phase 4 (final phase) of the ASL Recognizer project.

What this script does:
  1. Loads the trained model (sign_model.pkl) at startup.
  2. Opens the webcam and runs MediaPipe hand detection on every frame.
  3. Turns the detected hand into the same 63-number, wrist-normalized
     feature format train_model.py used, and asks the model for a
     predicted letter + confidence score.
  4. Uses a 20-frame "hold timer": a letter only counts as locked in once
     it has shown up in at least 18 of the last 20 frames AND the current
     frame's confidence is at least 85%. Locking in speaks the letter out
     loud (on a background thread, so the video never freezes) and clears
     the buffer so the same letter has to be held again to repeat.
  5. Displays the current letter, confidence, buffer fill, and the last 5
     spoken letters as a running "word strip" on screen at all times.
  6. Press Q to quit: releases the webcam and shuts the speech engine down.
"""

import os                     # Builds file paths and checks the model file exists
import threading               # Runs text-to-speech in the background, off the video thread
import queue                   # Hands letters from the video loop to the speech thread safely
from collections import deque  # Fixed-size sliding window for the last 20 predictions

import cv2               # OpenCV: webcam access, drawing, and displaying the video window
import mediapipe as mp    # Detects the hand and its 21 landmarks
import joblib             # Loads the trained classifier from sign_model.pkl
import pythoncom          # Required so pyttsx3's Windows speech engine works on a background thread (see run_tts_worker)
import pyttsx3            # Offline text-to-speech

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(SCRIPT_DIRECTORY, "sign_model.pkl")
HAND_MODEL_FILE_PATH = os.path.join(SCRIPT_DIRECTORY, "hand_landmarker.task")

NUMBER_OF_HAND_LANDMARKS = 21      # MediaPipe always reports exactly 21 points per hand
PREDICTION_BUFFER_SIZE = 20         # How many recent frames we remember
CONSISTENCY_COUNT_REQUIRED = 18     # A letter must appear this many times in the buffer to lock in
CONFIDENCE_LOCK_THRESHOLD = 0.85    # The current frame must be at least this confident to lock in
WORD_STRIP_LENGTH = 5               # How many recently-spoken letters to show on screen


# ---------------------------------------------------------------------------
# Landmark feature extraction (must match train_model.py exactly)
# ---------------------------------------------------------------------------

def flatten_hand_landmarks(hand_landmarks):
    """Converts MediaPipe's 21 separate landmark objects into one flat
    list of 63 raw numbers: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20].
    """
    flattened_coordinates = []
    for landmark in hand_landmarks:
        flattened_coordinates.append(landmark.x)
        flattened_coordinates.append(landmark.y)
        flattened_coordinates.append(landmark.z)
    return flattened_coordinates


def normalize_relative_to_wrist(flattened_coordinates):
    """Subtracts the wrist's (landmark 0) x/y/z from every landmark's
    x/y/z, turning absolute on-screen positions into "distance from the
    wrist" instead.

    This has to produce exactly the same numbers train_model.py computed
    when it normalized hand_data.csv before training. If this function
    did anything different -- a different reference point, a different
    order of operations -- the live features would not match the
    statistical pattern the model actually learned, and predictions would
    be meaningless even though nothing would "crash."
    """
    wrist_x = flattened_coordinates[0]
    wrist_y = flattened_coordinates[1]
    wrist_z = flattened_coordinates[2]

    normalized_coordinates = []
    for landmark_index in range(NUMBER_OF_HAND_LANDMARKS):
        base_index = landmark_index * 3
        normalized_coordinates.append(flattened_coordinates[base_index] - wrist_x)
        normalized_coordinates.append(flattened_coordinates[base_index + 1] - wrist_y)
        normalized_coordinates.append(flattened_coordinates[base_index + 2] - wrist_z)

    return normalized_coordinates


# ---------------------------------------------------------------------------
# Non-blocking text-to-speech
# ---------------------------------------------------------------------------

def run_tts_worker(speech_queue):
    """Runs forever on its own background thread: pulls letters out of
    speech_queue and speaks them one at a time, until it receives the
    special value None, which tells it to shut down.

    pythoncom.CoInitialize() is required here because pyttsx3's Windows
    speech engine (SAPI5) is built on COM, and COM must be initialized
    separately on every individual thread that uses it. Without this
    line, calling the engine from a background thread doesn't raise an
    error -- it just hangs forever on the very first speech request,
    which is a easy thing to miss until you actually test it.
    """
    pythoncom.CoInitialize()
    speech_engine = pyttsx3.init()

    while True:
        letter_to_speak = speech_queue.get()  # Blocks here until the video loop adds something
        if letter_to_speak is None:
            break
        speech_engine.say(letter_to_speak)
        speech_engine.runAndWait()

    speech_engine.stop()
    pythoncom.CoUninitialize()


# ---------------------------------------------------------------------------
# On-screen display
# ---------------------------------------------------------------------------

def draw_centered_text(video_frame, text, y_position, font_scale, color, thickness):
    """Draws text horizontally centered on the frame at a given height.

    We measure the text's pixel width with cv2.getTextSize first instead
    of using a fixed x position, because a narrow letter like "I" and a
    wide one like "W" would otherwise end up drifting left or right of
    center as the prediction changes from frame to frame.
    """
    frame_width = video_frame.shape[1]
    (text_width, _text_height), _baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x_position = (frame_width - text_width) // 2
    cv2.putText(video_frame, text, (x_position, y_position),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    """Loads the model, opens the webcam, and runs the live recognition
    loop until the user presses Q.
    """

    # ---- Load the trained model ----
    if not os.path.exists(MODEL_FILE_PATH):
        print(f"ERROR: could not find '{MODEL_FILE_PATH}'.")
        print("Run train_model.py first (Phase 3) to create sign_model.pkl.")
        return

    classifier = joblib.load(MODEL_FILE_PATH)
    print("Model loaded.")

    # ---- Set up MediaPipe's hand detector (same settings as collect_data.py) ----
    base_options = mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_FILE_PATH)
    hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        # IMAGE mode treats every frame as an independent photo, exactly
        # like collect_data.py did when the training samples were
        # captured -- using VIDEO mode's frame-to-frame smoothing here
        # instead would shift the landmark positions slightly compared
        # to what the model was trained on.
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    )
    hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_landmarker_options)
    drawing_utils = mp.tasks.vision.drawing_utils
    drawing_styles = mp.tasks.vision.drawing_styles
    hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    # ---- Start the background speech thread ----
    speech_queue = queue.Queue()
    speech_thread = threading.Thread(target=run_tts_worker, args=(speech_queue,), daemon=True)
    speech_thread.start()

    # ---- Open the webcam ----
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("ERROR: could not open the webcam. Is it being used by another app?")
        return

    prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
    spoken_letters_history = deque(maxlen=WORD_STRIP_LENGTH)

    print("Live recognition started. Press 'q' to quit.\n")

    while True:
        frame_was_read, video_frame = video_capture.read()
        if not frame_was_read:
            print("ERROR: failed to read a frame from the webcam.")
            break

        # Mirror the frame, matching collect_data.py exactly. The model
        # was trained on mirrored frames, so skipping this would feed it
        # a left-right reversed hand shape -- normalization alone does
        # not fix that, since a mirror image has genuinely different
        # relative landmark positions, not just a different location.
        video_frame = cv2.flip(video_frame, 1)

        rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = hand_landmarker.detect(mediapipe_image)

        hand_was_detected = len(detection_result.hand_landmarks) > 0

        if hand_was_detected:
            detected_hand_landmarks = detection_result.hand_landmarks[0]

            drawing_utils.draw_landmarks(
                video_frame,
                detected_hand_landmarks,
                hand_connections,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )

            raw_coordinates = flatten_hand_landmarks(detected_hand_landmarks)
            normalized_coordinates = normalize_relative_to_wrist(raw_coordinates)

            # predict_proba gives a probability for every letter at once,
            # so we can get both the predicted letter (highest probability)
            # and its confidence from a single forest pass.
            class_probabilities = classifier.predict_proba([normalized_coordinates])[0]
            best_class_index = max(range(len(class_probabilities)), key=lambda index: class_probabilities[index])
            predicted_letter = classifier.classes_[best_class_index]
            confidence = class_probabilities[best_class_index]

            prediction_buffer.append(predicted_letter)

            draw_centered_text(video_frame, str(predicted_letter), 80, 2.5, (255, 255, 255), 4)
            draw_centered_text(video_frame, f"Confidence: {confidence * 100:.1f}%", 130, 0.9, (255, 255, 255), 2)

            # ---- Hold-timer lock-in check ----
            if len(prediction_buffer) == PREDICTION_BUFFER_SIZE:
                matching_frame_count = prediction_buffer.count(predicted_letter)
                letter_is_consistent = matching_frame_count >= CONSISTENCY_COUNT_REQUIRED
                confidence_is_high_enough = confidence >= CONFIDENCE_LOCK_THRESHOLD

                if letter_is_consistent and confidence_is_high_enough:
                    speech_queue.put(predicted_letter)
                    spoken_letters_history.append(predicted_letter)
                    # Clearing the buffer is what stops the same held
                    # gesture from immediately re-triggering: it now has
                    # to build back up to 18/20 frames before locking in again.
                    prediction_buffer.clear()
        else:
            # A half-built streak of consistent frames doesn't mean
            # anything once the hand disappears, so we drop it rather
            # than let it carry over and mix with a different gesture later.
            prediction_buffer.clear()
            draw_centered_text(video_frame, "No hand detected", 80, 1.2, (0, 0, 255), 2)

        cv2.putText(video_frame, f"Buffer: {len(prediction_buffer)}/{PREDICTION_BUFFER_SIZE}",
                    (20, video_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        word_strip_text = "Word: " + "".join(str(letter) for letter in spoken_letters_history)
        cv2.putText(video_frame, word_strip_text, (20, video_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("ASL Live Recognizer", video_frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    # ---- Clean shutdown ----
    video_capture.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()

    speech_queue.put(None)          # Tells the speech thread to stop its loop
    speech_thread.join(timeout=3)   # Gives it a moment to finish speaking and shut down

    print("Webcam closed. Goodbye!")


if __name__ == "__main__":
    main()
