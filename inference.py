import numpy as np
import tensorflow as tf
import librosa
import sys

# Path to the TFLite model
tflite_model_path = './model/model.tflite'


def preprocess_audio(file_path, sample_rate=48000):
    data, sr = librosa.load(file_path, sr=sample_rate)
    data, _ = librosa.effects.trim(data, top_db=10)
    data = librosa.util.fix_length(data, size=24000)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20)
    # Reshape to match model input: (1, 47, 20)
    mfcc = mfcc[np.newaxis, :, :]
    mfcc = mfcc.transpose(0, 2, 1)
    return mfcc.astype(np.float32)


def run_inference(audio_file):
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess audio
    input_data = preprocess_audio(audio_file)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output, axis=1)[0]
    print(f'Predicted digit: {predicted_class}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python inference.py <audio_file.wav>')
        sys.exit(1)
    audio_file = sys.argv[1]
    run_inference(audio_file)
