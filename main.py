import argparse
import pickle
import time
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from typing import List, Tuple, Dict

# Constants
WINDOW_SIZE = 4
VECTOR_SIZE = 71

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Diacritic Restoration Tool')
parser.add_argument('--dataset', type=str, default="data/diac_corpus.txt", help='Path to the dataset')
parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
parser.add_argument('--train', action='store_true', help='Train the model')
parser.add_argument('--predict', action='store_true', help='Predict using the model')
parser.add_argument('--verbose', action='store_true', help='Verbose training output')


class Dataset:
    """
    A class for handling datasets related to diacritic restoration.
    """
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"
    ALL_DIA = "acdeinorstuyzáčďéěíňóřšťúůýž"
    dia_dict = {letter: ind for ind, letter in enumerate(ALL_DIA)}
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self, name: str = "data/diac_corpus.txt"):
        """
        Initialize Dataset with given corpus file name.
        """
        logging.info("Loading dataset...")
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

        with open("data/diacritics-etest.txt", "r", encoding="utf-8-sig") as dataset_file:
            self.gold = dataset_file.read()
        self.test = self.gold.translate(self.DIA_TO_NODIA)

        self.data_unique = sorted(list(set(self.data)))
        self.target_unique = sorted(list(set(self.target)))

        self.data_int_to_char, self.data_char_to_int = self._get_data_map(self.data_unique)
        self.windows_data = self._make_data_windows(self.data, WINDOW_SIZE, self.data_char_to_int)
        self.windows_target = self._make_target_windows(self.target, WINDOW_SIZE)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.windows_data, self.windows_target,
                                                                                test_size=0.2, random_state=42,
                                                                                shuffle=False)
        logging.info("Dataset loaded.")

    def _get_data_map(self, data_unique: List[str]) -> Tuple[Dict[int, str], Dict[str, int]]:
        data_int_to_char = {i: char for i, char in enumerate(data_unique)}
        data_char_to_int = {char: i for i, char in enumerate(data_unique)}
        return data_int_to_char, data_char_to_int

    def _make_data_windows(self, data: str, window_size: int, data_char_to_int: Dict[str, int]) -> np.array:
        windows_data = []
        for letter_ind in range(window_size, len(data) - window_size):
            vector = [0] * VECTOR_SIZE * (2 * window_size + 1)
            if data[letter_ind] in self.LETTERS_NODIA:
                neighborhood = data[letter_ind - window_size:letter_ind + window_size + 1]
                for offset, neighbor in enumerate(neighborhood):
                    vector[data_char_to_int.get(neighbor, 1) + offset * VECTOR_SIZE] = 1
                windows_data.append(vector)
        return np.array(windows_data)

    def _make_target_windows(self, target: str, window_size: int) -> np.array:
        windows_target = []
        for letter_ind in range(window_size, len(target) - window_size):
            if target[letter_ind] in self.ALL_DIA:
                windows_target.append(np.eye(len(self.ALL_DIA))[self.dia_dict[target[letter_ind]]])
        return np.array(windows_target)


class Model:
    """
    A class for training, loading, and making predictions using the MLP model for diacritic restoration.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.model = MLPClassifier(hidden_layer_sizes=100, max_iter=200, activation="relu", solver="adam",
                                   random_state=42, verbose=args.verbose, early_stopping=True)
        if args.train:
            self._train_model()
        else:
            self._load_model()

    def _train_model(self):
        logging.info("Training model...")
        start = time.time()
        self.model.fit(self.dataset.X_train, self.dataset.y_train)
        end = time.time()
        logging.info(f"Model trained in {end - start:.2f} seconds.")
        with open("models/model.pkl", "wb") as model_file:
            logging.info("Saving model...")
            pickle.dump(self.model, model_file)
            logging.info("Model saved.")

    def _load_model(self):
        with open("models/model.pkl", "rb") as model_file:
            logging.info("Loading model...")
            self.model = pickle.load(model_file)
            logging.info("Model loaded.")

    def predict(self, data: str) -> str:
        result = data[:WINDOW_SIZE]
        for i in range(WINDOW_SIZE, len(data) - WINDOW_SIZE):
            current_char = data[i]
            if current_char in Dataset.LETTERS_NODIA:
                window = self._make_windows(data, WINDOW_SIZE, self.dataset.data_char_to_int, i)
                predicted_vector = self.model.predict([window])[0]
                predicted_index = np.argmax(predicted_vector)
                prediction = Dataset.ALL_DIA[predicted_index]
                result += prediction
            else:
                result += current_char
        result += data[-WINDOW_SIZE:]
        return result

    def _make_windows(self, data: str, window_size: int, data_char_to_int: Dict[str, int], padding: int) -> List[int]:
        window = [0] * VECTOR_SIZE * (2 * window_size + 1)
        neighborhood = data[padding - window_size:padding + window_size + 1]
        for offset, neighbor in enumerate(neighborhood):
            window[data_char_to_int.get(neighbor, 1) + offset * VECTOR_SIZE] = 1
        return window

    def per_char_accuracy(self, gold: str, pred: str) -> float:
        correct = 0
        total_chars = 0
        for g, p in zip(gold, pred):
            if g != ' ':
                if g == p:
                    correct += 1
                total_chars += 1
        return correct / total_chars


def main():
    if not args.train and not args.evaluate and not args.predict:
        logging.error("No action specified. Use --train, --evaluate or --predict.")
        return

    dataset = Dataset(name=args.dataset)
    model = Model(dataset)

    if args.predict:
        input_text = input("Enter text without diacritics: ")
        print(model.predict(input_text))
    elif args.evaluate:
        predictions = model.predict(dataset.test)
        baseline_accuracy = model.per_char_accuracy(dataset.gold, dataset.test)
        accuracy = model.per_char_accuracy(dataset.gold, predictions)
        logging.info(f"Baseline accuracy: {(baseline_accuracy * 100):.3f}%")
        logging.info(f"Per char accuracy: {(accuracy * 100):.3f}%")
    elif args.train:
        training_accuracy = model.per_char_accuracy(dataset.target[WINDOW_SIZE:-WINDOW_SIZE],
                                                    model.predict(dataset.data))
        logging.info(f"Training accuracy: {(training_accuracy * 100):.3f}%")


if __name__ == "__main__":
    args = parser.parse_args()
    main()
