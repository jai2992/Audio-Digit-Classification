<div align="center">
  <h2>Audio Digit Classification</h2>

  ![Python](https://img.shields.io/badge/Python-3.x-blue)
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
  ![Librosa](https://img.shields.io/badge/Librosa-Audio-green)
  ![Status](https://img.shields.io/badge/Status-Active-brightgreen)

  <p>A simple and clean Deep Learning project to classify spoken digits using MFCC features and a Convolutional Neural Network (CNN).</p>
</div>

---

### 📝 Overview

This repository contains a complete pipeline for training an audio classification model on the Free Spoken Digit Dataset (FSDD). It processes audio files to extract Mel-Frequency Cepstral Coefficients (MFCCs) and trains a lightweight CNN model using TensorFlow/Keras. The final model is exported to TFLite format for deployment.

### ✨ Features

- **Audio Processing**: Automatic trimming and padding of audio files.
- **Feature Extraction**: extraction of MFCC features using `librosa`.
- **Deep Learning**: Custom CNN architecture for accurate classification.
- **Model Export**: Exports the trained model to `.tflite` for edge deployment.

### 📂 Project Structure

```
├── free-spoken-digit-dataset-master/
│   └── recordings/       # Contains .wav files for digits 0-9
├── notebook/             # Jupyter notebooks for exploration
├── main.py               # Main script for Training and Evaluation
├── requirements.txt      # Python dependencies
├── setup.sh              # Script to download dataset
└── README.md             # Project documentation
```

### 🚀 Setup & Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Audio-Digit-Classification
   ```

2. **Download the Dataset**
   Run the setup script to download and unzip the Free Spoken Digit Dataset.
   ```bash
   bash setup.sh
   ```
   *Note: If the `free-spoken-digit-dataset-master` folder is already present, you can skip this step.*

3. **Install Dependencies**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

### 🏃 Usage

To run the training pipeline, simply execute the `main.py` script:

```bash
python main.py
```

**What happens next?**
1.  The script reads audio files from `free-spoken-digit-dataset-master/recordings`.
2.  It processes the audio (trim silence, fix length) and extracts MFCCs.
3.  A dataset is created and split into training, validation, and test sets.
4.  A CNN model is trained for 50 epochs.
5.  The model's accuracy is evaluated on the test set.
6.  The trained model is saved as `model/model.tflite`.

### 📊 Model Architecture

The model is a Sequential CNN with the following layers:
-   Input Layer (MFCC features)
-   Conv2D + ReLU (32 filters)
-   Conv2D + ReLU (64 filters)
-   Global Average Pooling
-   Dense + ReLU
-   Output Dense (10 classes, Softmax)

### 🤝 Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---