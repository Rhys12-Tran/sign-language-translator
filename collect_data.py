"""
collect_data.py
-----------------
Data collection script for a ONE-HANDED sign language alphabet recognizer.

What this script does:
  1. Opens the webcam and mirrors it (so moving your hand right on screen
     matches moving your hand right in real life).
  2. Runs MediaPipe hand detection (1 hand max) and draws the 21 landmark
     dots on the detected hand in real time.
  3. Lets you record labeled training samples for the 26 letters A-Z by
     pressing SPACE. Each sample is one row of 63 numbers (21 landmarks x
     x/y/z) plus the letter label, appended to hand_data.csv.
  4. Is resumable: every time it starts, it reads how many samples already
     exist for each letter in hand_data.csv and continues counting up from
     there instead of starting over or overwriting old data.

Controls (shown in the terminal when the script starts):
  SPACE -> save the current hand position as one sample for the active letter
  N     -> move to the next letter
  B     -> move back to the previous letter
  Q     -> quit and print a summary of samples collected per letter
"""

import os    # Used to check whether hand_data.csv already exists
import csv   # Used to read/write hand_data.csv in a standard, spreadsheet-friendly format

import cv2              # OpenCV: webcam access, drawing on frames, showing the window
import mediapipe as mp   # MediaPipe: detects the hand and its 21 landmarks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHABET_LETTERS = [chr(ord("A") + offset) for offset in range(26)]

SAMPLES_PER_LETTER_TARGET = 100  # How many samples we want collected per letter

NUMBER_OF_HAND_LANDMARKS = 21  # MediaPipe always reports exactly 21 points per hand

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIRECTORY, "hand_data.csv")
HAND_MODEL_FILE_PATH = os.path.join(SCRIPT_DIRECTORY, "hand_landmarker.task")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def build_csv_header_row():
    """Builds the CSV header: x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label.

    This is generated with a loop rather than typed out by hand because
    there are 64 columns total -- a loop can't accidentally get one of
    them out of order or miscounted the way manually typed text could.
    """
    header_row = []
    for landmark_index in range(NUMBER_OF_HAND_LANDMARKS):
        header_row.append(f"x{landmark_index}")
        header_row.append(f"y{landmark_index}")
        header_row.append(f"z{landmark_index}")
    header_row.append("label")
    return header_row


def ensure_csv_file_exists_with_header():
    """Creates hand_data.csv with a header row the first time this script
    runs, and does nothing if the file already exists.

    We only ever want the header written once. If we wrote it every time
    the script started, every session after the first would add a second
    (and third, and fourth...) header row mixed in with real data rows,
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
    every letter even though real samples already exist, and the user
    could keep collecting well past the target without realizing it.
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


def save_sample_to_csv(flattened_landmark_coordinates, letter_label):
    """Appends one labeled sample (63 numbers + a letter) as a new row at
    the end of hand_data.csv.

    Opening in append ("a") mode instead of write ("w") mode matters here:
    "w" mode erases the entire file before writing, which would destroy
    every sample collected in previous sessions and break resumability.
    """
    with open(CSV_FILE_PATH, "a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(flattened_landmark_coordinates + [letter_label])


# ---------------------------------------------------------------------------
# MediaPipe / landmark helpers
# ---------------------------------------------------------------------------

def flatten_hand_landmarks(hand_landmarks):
    """Converts MediaPipe's 21 separate landmark objects into one flat list
    of 63 numbers: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20].

    A flat list is what the CSV writer (and later the scikit-learn
    classifier) expects -- they work with plain rows of numbers, not a
    nested structure of 21 separate landmark objects.
    """
    flattened_coordinates = []
    for landmark in hand_landmarks:
        flattened_coordinates.append(landmark.x)
        flattened_coordinates.append(landmark.y)
        flattened_coordinates.append(landmark.z)
    return flattened_coordinates


def find_starting_letter_index(sample_counts_by_letter):
    """Picks which letter to start on: the first letter (A through Z) that
    does not yet have a full set of samples.

    Without this, every new session would always start back at "A", even
    if earlier sessions already finished A through M -- this is what makes
    "continues from where it left off" actually true instead of just true
    for the CSV file alone.
    """
    for letter_index, letter in enumerate(ALPHABET_LETTERS):
        if sample_counts_by_letter[letter] < SAMPLES_PER_LETTER_TARGET:
            return letter_index
    return len(ALPHABET_LETTERS) - 1  # Every letter already has enough samples


# ---------------------------------------------------------------------------
# On-screen display
# ---------------------------------------------------------------------------

def draw_status_overlay(video_frame, current_letter, samples_collected_for_letter, hand_was_detected):
    """Draws the current letter, the sample count, hand-detection status,
    and a COMPLETE message (once the target is reached) on top of the
    video frame.

    Keeping all the cv2.putText calls in one function -- instead of
    scattering them through the main loop -- means the main loop only has
    to read the camera and react to key presses, which is easier to follow.
    """
    cv2.putText(video_frame, f"Letter: {current_letter}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.putText(video_frame,
                f"Samples: {samples_collected_for_letter} / {SAMPLES_PER_LETTER_TARGET}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Green for "good to go", red for "nothing to save right now" -- a
    # color difference is something you notice in your peripheral vision
    # while focused on getting your hand shape right, a text difference is not.
    if hand_was_detected:
        cv2.putText(video_frame, "HAND DETECTED", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(video_frame, "NO HAND DETECTED", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if samples_collected_for_letter >= SAMPLES_PER_LETTER_TARGET:
        cv2.putText(video_frame, "COMPLETE!", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)


def print_collection_summary(sample_counts_by_letter):
    """Prints a per-letter breakdown of how many samples were collected,
    used when the user quits with Q.

    A per-letter breakdown (rather than one combined total) is what lets
    the user see at a glance which specific letters still need more data
    before moving on to training a classifier.
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
    webcam, then loops reading frames, detecting the hand, drawing the
    overlay, and reacting to key presses until the user quits with Q.
    """
    ensure_csv_file_exists_with_header()
    sample_counts_by_letter = count_existing_samples_per_letter()
    current_letter_index = find_starting_letter_index(sample_counts_by_letter)

    base_options = mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_FILE_PATH)
    hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        # IMAGE mode (rather than VIDEO mode) treats every frame as an
        # independent still photo. We want that here -- each saved sample
        # should reflect exactly what is in front of the camera at the
        # moment SPACE is pressed, not a smoothed estimate blended with
        # earlier frames.
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
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
        # is the orientation people naturally expect when watching
        # themselves move on screen.
        video_frame = cv2.flip(video_frame, 1)

        # MediaPipe expects RGB, OpenCV gives us BGR.
        rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = hand_landmarker.detect(mediapipe_image)

        hand_was_detected = len(detection_result.hand_landmarks) > 0
        detected_hand_landmarks = detection_result.hand_landmarks[0] if hand_was_detected else None

        if hand_was_detected:
            drawing_utils.draw_landmarks(
                video_frame,
                detected_hand_landmarks,
                hand_connections,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )

        current_letter = ALPHABET_LETTERS[current_letter_index]
        draw_status_overlay(
            video_frame,
            current_letter,
            sample_counts_by_letter[current_letter],
            hand_was_detected,
        )

        cv2.imshow("ASL Data Collector", video_frame)

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
            if hand_was_detected:
                flattened_coordinates = flatten_hand_landmarks(detected_hand_landmarks)
                save_sample_to_csv(flattened_coordinates, current_letter)
                sample_counts_by_letter[current_letter] += 1
            else:
                # Saving a row of meaningless numbers would quietly poison
                # the training data later, so we refuse and tell the user why.
                # The red "NO HAND DETECTED" status above is the on-screen
                # half of this warning; this terminal line is the other half.
                print("WARNING: no hand detected -- sample not saved.")

    video_capture.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()

    print_collection_summary(sample_counts_by_letter)
    print("Webcam closed. Goodbye!")


if __name__ == "__main__":
    main()
