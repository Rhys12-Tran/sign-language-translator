"""
collect_data.py
-----------------
Data collection script for a TWO-HANDED New Zealand Sign Language (NZSL)
alphabet recognizer.

What this script does:
  1. Opens the webcam and mirrors it (so moving your hand right on screen
     matches moving your hand right in real life -- this also matters for
     handedness labeling, see draw_hand_labels() below).
  2. Runs MediaPipe hand detection with up to 2 hands per frame and draws
     the 21 landmark dots on every hand it finds.
  3. Lets you record labeled training samples for the 26 letters A-Z by
     pressing SPACE. Each sample is one row of 126 numbers -- the left
     hand's 21 landmarks (x/y/z) followed by the right hand's 21 landmarks
     (x/y/z) -- plus the letter label, appended to hand_data.csv.
  4. If only one hand is visible, the missing hand's 63 values are filled
     with zeros instead of skipping the frame, so two-handed signs can
     still be partially recorded.
  5. Is resumable: every time it starts, it reads how many samples already
     exist for each letter in hand_data.csv and continues counting up from
     there instead of starting over or overwriting old data.

Controls (shown in the terminal when the script starts):
  SPACE -> save the current hand position(s) as one sample for the active letter
  N     -> move to the next letter
  B     -> move back to the previous letter
  Q     -> quit and print a summary of samples collected per letter
"""

import os    # Used to check whether hand_data.csv already exists
import csv   # Used to read/write hand_data.csv in a standard, spreadsheet-friendly format

import cv2              # OpenCV: webcam access, drawing on frames, showing the window
import mediapipe as mp   # MediaPipe: detects hands and their 21 landmarks each

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHABET_LETTERS = [chr(ord("A") + offset) for offset in range(26)]

SAMPLES_PER_LETTER_TARGET = 100  # How many samples we want collected per letter

NUMBER_OF_HAND_LANDMARKS = 21              # MediaPipe always reports 21 points per hand
VALUES_PER_HAND = NUMBER_OF_HAND_LANDMARKS * 3   # 21 landmarks x (x, y, z) = 63 numbers
HAND_SIDES = ("left", "right")             # Fixed column order: left hand first, then right hand

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIRECTORY, "hand_data.csv")
HAND_MODEL_FILE_PATH = os.path.join(SCRIPT_DIRECTORY, "hand_landmarker.task")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def build_csv_header_row():
    """Builds the 126-feature CSV header: left_x0,left_y0,left_z0, ...,
    left_x20,left_y20,left_z20, then the same 63 columns again prefixed
    "right_", then "label" -- 127 columns total.

    Every column is prefixed with which hand it belongs to instead of
    reusing x0,y0,z0 for both hands, because two columns sharing the same
    name would make the CSV ambiguous to read back correctly later.
    """
    header_row = []
    for hand_side in HAND_SIDES:
        for landmark_index in range(NUMBER_OF_HAND_LANDMARKS):
            header_row.append(f"{hand_side}_x{landmark_index}")
            header_row.append(f"{hand_side}_y{landmark_index}")
            header_row.append(f"{hand_side}_z{landmark_index}")
    header_row.append("label")
    return header_row


def ensure_csv_file_exists_with_header():
    """Creates hand_data.csv with a header row the first time this script
    runs, and does nothing if the file already exists.

    We only ever want the header written once -- writing it again on
    every run would mix extra header rows in among the real data rows,
    which would break training later.
    """
    if not os.path.exists(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(build_csv_header_row())


def count_existing_samples_per_letter():
    """Reads hand_data.csv and counts how many rows already exist for each
    letter, returning a dict like {"A": 100, "B": 37, "C": 0, ...}.

    This is what makes the script resumable: without re-counting what is
    already saved on disk, restarting the script would show "0 / 100" for
    every letter even though real samples already exist.
    """
    sample_counts_by_letter = {letter: 0 for letter in ALPHABET_LETTERS}

    with open(CSV_FILE_PATH, "r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)  # Skip the header row -- it is not a sample
        for data_row in csv_reader:
            if not data_row:
                continue
            letter_label = data_row[-1]  # The label is always the last column
            if letter_label in sample_counts_by_letter:
                sample_counts_by_letter[letter_label] += 1

    return sample_counts_by_letter


def save_sample_to_csv(flattened_two_hand_coordinates, letter_label):
    """Appends one labeled sample (126 numbers + a letter) as a new row at
    the end of hand_data.csv.

    Opening in append ("a") mode instead of write ("w") mode matters:
    "w" mode erases the entire file first, which would destroy every
    sample collected in previous sessions and break resumability.
    """
    with open(CSV_FILE_PATH, "a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(flattened_two_hand_coordinates + [letter_label])


# ---------------------------------------------------------------------------
# MediaPipe / landmark helpers
# ---------------------------------------------------------------------------

def flatten_hand_landmarks(hand_landmarks):
    """Converts one hand's 21 separate landmark objects into one flat list
    of 63 numbers: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20].
    """
    flattened_coordinates = []
    for landmark in hand_landmarks:
        flattened_coordinates.append(landmark.x)
        flattened_coordinates.append(landmark.y)
        flattened_coordinates.append(landmark.z)
    return flattened_coordinates


def build_two_hand_feature_vector(detection_result):
    """Combines whichever hands MediaPipe found this frame into one fixed
    126-number vector: the first 63 numbers are always the LEFT hand and
    the last 63 are always the RIGHT hand, using MediaPipe's own
    handedness classification to decide which detected hand goes where.

    If a hand is missing, its 63 slots stay zero instead of the frame
    being skipped entirely -- this is what lets a one-handed moment of a
    two-handed sign still be recorded, per the project's requirements.

    Returns (feature_vector, hands_detected_count).
    """
    left_hand_values = [0.0] * VALUES_PER_HAND
    right_hand_values = [0.0] * VALUES_PER_HAND

    for hand_index, hand_landmarks in enumerate(detection_result.hand_landmarks):
        handedness_label = detection_result.handedness[hand_index][0].category_name
        flattened_coordinates = flatten_hand_landmarks(hand_landmarks)

        if handedness_label == "Left":
            left_hand_values = flattened_coordinates
        elif handedness_label == "Right":
            right_hand_values = flattened_coordinates

    hands_detected_count = len(detection_result.hand_landmarks)
    return left_hand_values + right_hand_values, hands_detected_count


def find_starting_letter_index(sample_counts_by_letter):
    """Picks which letter to start on: the first letter (A through Z) that
    does not yet have a full set of samples, so resuming a session lands
    you back where you left off instead of always restarting at "A".
    """
    for letter_index, letter in enumerate(ALPHABET_LETTERS):
        if sample_counts_by_letter[letter] < SAMPLES_PER_LETTER_TARGET:
            return letter_index
    return len(ALPHABET_LETTERS) - 1  # Every letter already has enough samples


# ---------------------------------------------------------------------------
# On-screen display
# ---------------------------------------------------------------------------

def draw_hand_labels(video_frame, detection_result):
    """Draws a "Left" or "Right" label next to each detected hand's
    wrist, using MediaPipe's handedness classification rather than each
    hand's position on screen -- position-based guessing would break the
    instant two hands cross over each other.

    MediaPipe's handedness model assumes the input image is mirrored
    (i.e. a front-facing/selfie-style view), which is exactly what we
    feed it since the frame is flipped in main() before detection runs.
    That flip is what makes "Left" on screen actually mean the signer's
    own left hand, instead of being swapped.
    """
    frame_height, frame_width = video_frame.shape[:2]

    for hand_index, hand_landmarks in enumerate(detection_result.hand_landmarks):
        handedness_label = detection_result.handedness[hand_index][0].category_name
        wrist_landmark = hand_landmarks[0]
        wrist_pixel_x = int(wrist_landmark.x * frame_width)
        wrist_pixel_y = int(wrist_landmark.y * frame_height)

        label_color = (0, 255, 0) if handedness_label == "Left" else (0, 165, 255)
        cv2.putText(video_frame, handedness_label, (wrist_pixel_x - 20, wrist_pixel_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)


def draw_status_overlay(video_frame, current_letter, samples_collected_for_letter,
                         hands_detected_count):
    """Draws the current letter, the sample count, how many hands are
    currently visible, and a COMPLETE message (once the target is
    reached) on top of the video frame.
    """
    cv2.putText(video_frame, f"Letter: {current_letter}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.putText(video_frame,
                f"Samples: {samples_collected_for_letter} / {SAMPLES_PER_LETTER_TARGET}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Green once both hands are visible, yellow for one hand (still
    # savable, just zero-filled on one side), red for none (not savable)
    # -- a color difference reads at a glance while you're focused on
    # getting the hand shape right, a sentence would not.
    if hands_detected_count == 2:
        status_text, status_color = "Hands: 2/2 DETECTED", (0, 255, 0)
    elif hands_detected_count == 1:
        status_text, status_color = "Hands: 1/2 DETECTED", (0, 255, 255)
    else:
        status_text, status_color = "Hands: NOT DETECTED", (0, 0, 255)
    cv2.putText(video_frame, status_text, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    if samples_collected_for_letter >= SAMPLES_PER_LETTER_TARGET:
        cv2.putText(video_frame, "COMPLETE!", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)


def print_collection_summary(sample_counts_by_letter):
    """Prints a per-letter breakdown of how many samples were collected,
    used when the user quits with Q.
    """
    print("\n--- Data Collection Summary ---")
    for letter in ALPHABET_LETTERS:
        count = sample_counts_by_letter[letter]
        status = "OK" if count >= SAMPLES_PER_LETTER_TARGET else "needs more"
        print(f"  {letter}: {count} / {SAMPLES_PER_LETTER_TARGET}  ({status})")
    print("--------------------------------\n")


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    """Runs the full data collection session: sets up MediaPipe and the
    webcam, then loops reading frames, detecting up to two hands, drawing
    the overlay, and reacting to key presses until the user quits with Q.
    """
    ensure_csv_file_exists_with_header()
    sample_counts_by_letter = count_existing_samples_per_letter()
    current_letter_index = find_starting_letter_index(sample_counts_by_letter)

    base_options = mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_FILE_PATH)
    hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        # IMAGE mode treats every frame as an independent still photo,
        # which is what we want here -- each saved sample should reflect
        # exactly what is in front of the camera at the moment SPACE is
        # pressed, not a smoothed estimate blended with earlier frames.
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,  # NZSL's two-handed alphabet needs both hands tracked at once
    )
    hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_landmarker_options)

    drawing_utils = mp.tasks.vision.drawing_utils
    drawing_styles = mp.tasks.vision.drawing_styles
    hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("ERROR: could not open the webcam. Is it being used by another app?")
        return

    print("Data collection started.")
    print("Controls: SPACE = save sample | N = next letter | B = previous letter | Q = quit\n")

    while True:
        frame_was_read, video_frame = video_capture.read()
        if not frame_was_read:
            print("ERROR: failed to read a frame from the webcam.")
            break

        # Flipping horizontally makes the feed behave like a mirror, which
        # is both the orientation people expect when watching themselves
        # on screen AND required for MediaPipe's handedness labels (see
        # draw_hand_labels()) to match the signer's real left/right hand.
        video_frame = cv2.flip(video_frame, 1)

        rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = hand_landmarker.detect(mediapipe_image)

        for hand_landmarks in detection_result.hand_landmarks:
            drawing_utils.draw_landmarks(
                video_frame,
                hand_landmarks,
                hand_connections,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )
        draw_hand_labels(video_frame, detection_result)

        feature_vector, hands_detected_count = build_two_hand_feature_vector(detection_result)

        current_letter = ALPHABET_LETTERS[current_letter_index]
        draw_status_overlay(
            video_frame,
            current_letter,
            sample_counts_by_letter[current_letter],
            hands_detected_count,
        )

        cv2.imshow("NZSL Data Collector", video_frame)

        key_pressed = cv2.waitKey(1) & 0xFF

        if key_pressed in (ord("q"), ord("Q")):
            break

        elif key_pressed in (ord("n"), ord("N")):
            # Clamping at the last index (instead of wrapping back to "A")
            # avoids accidentally mixing new "A" samples in with a finished
            # A-Z pass just because someone held N down too long.
            current_letter_index = min(current_letter_index + 1, len(ALPHABET_LETTERS) - 1)

        elif key_pressed in (ord("b"), ord("B")):
            current_letter_index = max(current_letter_index - 1, 0)

        elif key_pressed == ord(" "):
            if hands_detected_count >= 1:
                save_sample_to_csv(feature_vector, current_letter)
                sample_counts_by_letter[current_letter] += 1
            else:
                # Saving an all-zero row would quietly poison the training
                # data later, so we refuse and tell the user why.
                print("WARNING: no hand detected -- sample not saved.")

    video_capture.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()

    print_collection_summary(sample_counts_by_letter)
    print("Webcam closed. Goodbye!")


if __name__ == "__main__":
    main()
