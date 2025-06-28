# Convolutional Bloc – A Hand-Rolled C++ Convolutional Feature Extractor for RGB and Greyscale Object Detection

This repository contains an archived implementation of a forward-only convolutional neural network engine written entirely in C++. It was developed as a personal exercise to deepen understanding of spatial feature processing, image-based object detection, and systems-level software design. I am yet to implement kernel-learning.

---

## Overview

The engine uses templated data structures and a modular architecture for basic image feature extraction and object classification (currently forward pass only). It includes:

- Modular convolutional and pooling layers
- A templated Image class
- A lightweight `FeatureMap` class to handle intermediate data
- A simplified training and inference loop for demonstration
- Header-only support for composability

The focus throughout was on clarity, minimalism, and precision in memory layout and data flow. No external machine learning frameworks were used.

---

## Project Structure

- `kernel.cpp/.h` – Convolution layer implementation
- `feature_map.h` – Core multidimensional feature tensor structure
- `main.cpp` – Prototype training script
- `sample.h` – Templated Image class using inheritance 

---

## Purpose

This project was created to build a practical intuition for convolutional architectures and to improve C++ fluency. It is not intended for production use or active development, and exists here for demonstration and archival purposes.

---

## Author

Samuel Clucas  
Durham University (BSc, Biological Sciences, First Class)  
Incoming MRes student, Biomedical Data Science – Imperial College London  
