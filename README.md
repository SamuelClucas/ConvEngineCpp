# ConvEngineCpp – A Hand-Rolled Convolutional Neural Network in C++

This repository contains an archived implementation of a forward-only convolutional neural network engine written entirely in C++. It was developed as a personal exercise to deepen understanding of spatial feature processing, image-based object detection, and systems-level software design.

---

## Overview

The engine uses templated data structures and a modular architecture to carry out basic image feature extraction and object classification. It includes:

- Modular convolutional and pooling layers
- A lightweight `FeatureMap` class to handle intermediate data
- A simplified training and inference loop for demonstration
- Custom polygon-based object detection heads
- Header-only support for composability

The focus throughout was on clarity, minimalism, and precision in memory layout and data flow. No external machine learning frameworks were used.

---

## Project Structure

- `kernel.cpp/.h` – Convolution layer implementation
- `feature_map.h` – Core multidimensional feature tensor structure
- `object_detector.cpp/.h` – Detection pipeline and polygon prediction
- `train_refactor.cpp` – Prototype training script
- `sample.h` – Basic input/output sample data types

---

## Purpose

This project was created to build a practical intuition for convolutional architectures and to improve C++ fluency. It is not intended for production use or active development, and exists here for demonstration and archival purposes.

---

## Author

Samuel Clucas  
Durham University (BSc, Biological Sciences, First Class)  
Incoming MRes student, Biomedical Data Science – Imperial College London  
