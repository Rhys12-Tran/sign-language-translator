"""
train_model.py
-----------------
Phase 3 of the ASL Recognizer project.

What this script does, step by step:
  1. Loads the hand landmark samples collected in Phase 2 (hand_data.csv).
  2. Splits each row into "features" (the 63 numbers describing the hand
     shape) and "target" (the letter that hand shape represents), then
     normalizes the features relative to the wrist landmark so the model
     learns the hand's SHAPE rather than where it happened to be sitting
     in the camera frame during collection.
  3. Splits the data into a training set (80%) and a test set (20%).
     The model only ever learns from the training set; the test set is
     held back so we can honestly check how well it does on hand shapes
     it has never seen before.
  4. Trains a Random Forest -- a model made of many small decision trees
     that each vote on the answer, which tends to work well on this kind
     of "a bunch of numbers describing a shape" data.
  5. Checks the model's accuracy on the test set.
  6. Prints a full classification report (precision/recall for every
     letter) and the letter pairs the model mixes up the most.
  7. If the model is good enough (70%+ accuracy), saves it to
     sign_model.pkl so Phase 4 (the live recognizer) can load and use it.
     Otherwise, it warns you instead of saving a model that isn't ready.
"""

import os  # Used to check whether hand_data.csv exists and to build file paths

import pandas as pd     # Loads and organizes the CSV data into a table (DataFrame)
import joblib            # Saves (and later loads) the trained model to/from a file

from sklearn.model_selection import train_test_split        # Splits data into train/test sets
from sklearn.ensemble import RandomForestClassifier          # The model we are training
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

# Building paths from this script's own folder means it works no matter
# which directory you happen to run it from.
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIRECTORY, "hand_data.csv")
MODEL_OUTPUT_PATH = os.path.join(SCRIPT_DIRECTORY, "sign_model.pkl")

ACCURACY_THRESHOLD = 0.70  # Minimum test accuracy required before we trust this model
NUMBER_OF_HAND_LANDMARKS = 21  # MediaPipe always reports exactly 21 points per hand


def normalize_features_relative_to_wrist(features_dataframe):
    """Subtracts the wrist's (landmark 0) x/y/z from every landmark's
    x/y/z, for every row in the DataFrame at once.

    Raw MediaPipe coordinates are normalized to the camera frame (0.0-1.0
    across the image), which means they still partly encode WHERE in the
    frame your hand was sitting, not just its shape. Making every landmark
    relative to the wrist removes that position dependency, so a letter
    looks the same to the model whether you sign it near the left edge of
    the frame or dead center. live_predict.py must apply this exact same
    transformation to live frames, or the model will be comparing
    differently-shaped numbers to what it learned during training.
    """
    normalized_dataframe = features_dataframe.copy()
    wrist_x = features_dataframe["x0"]
    wrist_y = features_dataframe["y0"]
    wrist_z = features_dataframe["z0"]

    for landmark_index in range(NUMBER_OF_HAND_LANDMARKS):
        normalized_dataframe[f"x{landmark_index}"] = features_dataframe[f"x{landmark_index}"] - wrist_x
        normalized_dataframe[f"y{landmark_index}"] = features_dataframe[f"y{landmark_index}"] - wrist_y
        normalized_dataframe[f"z{landmark_index}"] = features_dataframe[f"z{landmark_index}"] - wrist_z

    return normalized_dataframe


def main():
    """Runs the full training pipeline: load data, split it, train the
    model, evaluate it, and save it if it's good enough.
    """

    # -----------------------------------------------------------------
    # STEP 1: Load the data collected in Phase 2
    # -----------------------------------------------------------------
    if not os.path.exists(CSV_FILE_PATH):
        # Without this check, pandas would raise a confusing internal
        # error -- this message tells the user exactly what to do instead.
        print(f"ERROR: could not find '{CSV_FILE_PATH}'.")
        print("Run collect_data.py first (Phase 2) to create hand_data.csv.")
        return

    hand_data = pd.read_csv(CSV_FILE_PATH)
    print(f"Loaded {len(hand_data)} samples from hand_data.csv")

    # -----------------------------------------------------------------
    # STEP 2: Split into features (the hand shape numbers) and target (the letter)
    # -----------------------------------------------------------------
    # "label" is the letter column; every other column is one of the 63
    # x/y/z landmark coordinates that describe the hand shape.
    feature_columns = [column for column in hand_data.columns if column != "label"]
    features = hand_data[feature_columns]
    target_letters = hand_data["label"]

    # Normalize relative to the wrist (see the function's docstring for
    # why), then convert to a plain numpy array before training. Fitting
    # on a numpy array (instead of a DataFrame) means the saved model does
    # not remember pandas column names, so live_predict.py can hand it a
    # plain list of 63 numbers at inference time without any mismatch
    # warnings or errors.
    normalized_features = normalize_features_relative_to_wrist(features)
    feature_array = normalized_features.to_numpy()

    # -----------------------------------------------------------------
    # STEP 3: Split into training data (80%) and testing data (20%)
    # -----------------------------------------------------------------
    # random_state=42 makes the split repeatable -- running this script
    # again will always produce the exact same train/test split, which
    # makes it possible to fairly compare results between runs.
    training_features, testing_features, training_letters, testing_letters = train_test_split(
        feature_array, target_letters, test_size=0.2, random_state=42
    )

    # -----------------------------------------------------------------
    # STEP 4: Train the Random Forest classifier
    # -----------------------------------------------------------------
    # n_estimators=100 means the forest is made of 100 individual decision
    # trees; each tree votes on the predicted letter, and the majority
    # vote wins. More trees generally means steadier (less random)
    # predictions, at the cost of a bit more training time.
    print("Training Random Forest classifier...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(training_features, training_letters)

    # -----------------------------------------------------------------
    # STEP 5: Evaluate accuracy on the held-out test set
    # -----------------------------------------------------------------
    predicted_letters = classifier.predict(testing_features)
    test_accuracy = accuracy_score(testing_letters, predicted_letters)

    print("\n" + "=" * 40)
    print(f"TEST ACCURACY: {test_accuracy * 100:.2f}%")
    print("=" * 40 + "\n")

    # -----------------------------------------------------------------
    # STEP 6: Print a full classification report and the most confused pairs
    # -----------------------------------------------------------------
    # The classification report shows, for every letter: precision (when
    # the model guessed this letter, how often was it right?) and recall
    # (out of all the real examples of this letter, how many did it catch?).
    print("Classification Report (per letter):")
    print(classification_report(testing_letters, predicted_letters, zero_division=0))

    sorted_letters = sorted(target_letters.unique())
    confusion_grid = confusion_matrix(testing_letters, predicted_letters, labels=sorted_letters)

    # The confusion matrix's diagonal is correct predictions; everything
    # off the diagonal is a mistake. We add cell [i, j] and [j, i] together
    # so "A predicted as B" and "B predicted as A" count as one mix-up
    # between the same pair of letters, which is easier to read.
    confused_pairs = []
    for row_index in range(len(sorted_letters)):
        for column_index in range(row_index + 1, len(sorted_letters)):
            mix_up_count = confusion_grid[row_index, column_index] + confusion_grid[column_index, row_index]
            if mix_up_count > 0:
                confused_pairs.append((mix_up_count, sorted_letters[row_index], sorted_letters[column_index]))

    confused_pairs.sort(key=lambda pair_entry: pair_entry[0], reverse=True)

    print("Top 5 most confused letter pairs:")
    if not confused_pairs:
        print("  None -- the model made no mistakes on the test set.")
    else:
        for mix_up_count, letter_one, letter_two in confused_pairs[:5]:
            print(f"  {letter_one} <-> {letter_two}: confused {mix_up_count} time(s)")
    print()

    # -----------------------------------------------------------------
    # STEP 7: Save the model only if it's accurate enough to be useful
    # -----------------------------------------------------------------
    if test_accuracy >= ACCURACY_THRESHOLD:
        joblib.dump(classifier, MODEL_OUTPUT_PATH)
        print("Model saved.")
    else:
        print("Accuracy too low. Do NOT proceed to Phase 4. Recollect data with more variety.")


if __name__ == "__main__":
    main()
