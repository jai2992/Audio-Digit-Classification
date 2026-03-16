#!/bin/bash

echo "Downloading the dataset..."

wget -q https://github.com/jai2992/Audio-Digit-Classification/releases/download/dataset/archive.zip

echo "Downloaded successfully!"

unzip -q archive.zip
rm archive.zip

echo "./free-spoken-digit-dataset-master/ - dataset folder"
