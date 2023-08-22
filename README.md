# Diacritic Restoration Tool

This tool aims to restore diacritics to text using a neural network classifier. It's beneficial for languages where diacritics play a vital role in conveying the correct meaning of words.

## Prerequisites
Ensure you have Python 3.7+ installed on your system.

## Setup

1. Clone the repository:
    ```git clone https://github.com/rasimgil/DiacriticsRestorer.git```
    ```cd DiacriticsRestorer```

2. Set up a virtual environment:
```python -m venv venv```

3. Activate the virtual environment:

    - On Windows:
    ```.\venv\Scripts\activate```
    
    - On macOS and Linux:
    ```source venv/bin/activate```

4. Install the required packages:
```pip install -r requirements.txt```

## Usage

1. Training the model:
```python main.py --train```

1. Evaluating the model's accuracy:
```python main.py --evaluate```

1. Predicting diacritics for a provided input:
```python main.py --predict```

During prediction, you'll be prompted to enter text without diacritics, and the program will return the text with restored diacritics.

## Options

1. `--dataset <path_to_dataset>`: Specify a different dataset file.
2. `--verbose`: Get verbose training output.
