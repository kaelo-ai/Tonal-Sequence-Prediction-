# Tonal Sequence Prediction for Sesotho

## Description

This project develops a deep learning model to predict tonal sequences in Sesotho, a Bantu language where tone distinguishes meanings between words or syllables. Utilising a bi-directional Long Short-Term Memory (LSTM) network, the model leverages Mel-Frequency Cepstral Coefficients (MFCCs), vowel length, and duration information to accurately predict tones, enhancing applications such as speech recognition and text-to-speech synthesis.

## Installation

### Prerequisites

- Python 3.6 or higher
- CUDA-enabled GPU (optional, for faster training)

### Required Libraries

Install the necessary libraries using `pip`:

```bash
pip install torch torchaudio numpy pandas scikit-learn librosa python-Levenshtein
```


## Usage

### Data Preprocessing

1. **Phonetisation**: Convert Sesotho text to phonemes using the `phonetize_text` function.
2. **Vowel Extraction**: Identify and record vowel positions with `extract_vowel_positions`.
3. **Feature Extraction**: Align phonemes to audio features and extract MFCCs, vowel length, and duration.

### Training the Model

Run the training script:

bash

Copy code

`python train.py --data_path path/to/data --epochs 30 --batch_size 16 --learning_rate 0.001`

### Making Predictions

Use the trained model to predict tones on new audio files:

bash

Copy code

`python predict.py --model_path path/to/best_model.pth --audio_file path/to/audio.wav`

## Project Structure

- `train.py`: Script to train the bi-directional LSTM model.
- `predict.py`: Script to make tone predictions on new audio files.
- `model.py`: Defines the LSTM-based neural network architecture.
- `preprocess.py`: Contains functions for data preprocessing and feature extraction.
- `requirements.txt`: Lists all required Python libraries.
- `data/`: Directory containing training, validation, and test datasets.
- `models/`: Directory to save trained models.

## Training Output

During training, the model outputs training and validation loss, accuracy, and Character Error Rate (CER) for each epoch. The best model based on validation accuracy is saved automatically.

Example:

plaintext

Copy code

`Epoch 1 Results: Train Loss: 0.6930, Train Accuracy: 52.67% Val Loss: 0.6932, Val Accuracy: 50.68% Character Error Rate (CER): 44.29% Saving new best model with Validation Accuracy: 50.68%`

## Evaluation

After training, evaluate the model's performance on test data to assess accuracy and CER.
