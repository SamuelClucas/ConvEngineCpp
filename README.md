# Convolutional Bloc – A Hand-Rolled C++ Convolutional Feature Extractor for RGB and Greyscale Object Detection

This repository contains an archived implementation of a forward-only convolutional neural network engine written entirely in C++. It was developed as a personal exercise to deepen my understanding of spatial feature processing, image-based object detection, and systems-level software design.

I began designing the `Image` class using inheritance, but gradually transitioned to a compositional approach when developing the convolutional block architecture. Kernel learning is not yet implemented.

---

## Overview

The engine uses templated data structures and a modular architecture for basic image feature extraction and object classification (forward pass only). It includes:

- Modular convolution and pooling layers  
- A templated `Image` class  
- A lightweight `FeatureMap` for intermediate data  
- A simplified training and inference loop for demonstration  
- Header-only support for composability

The focus throughout was on clarity, minimalism, and precision in memory layout and data flow. No external machine learning frameworks were used.

---

## Project Structure

- `dataset.h` – Abstracted dataset loader  
- `sample.h` – Templated `Image` class (inheritance-based)  
- `label.h` – Label support for samples within datasets  

- `kernel.cpp/.h` – Convolutional layer logic  
- `feature_map.h` – Core multidimensional tensor structure  

- `bloc.h` – The convolutional block class: defines a flattened relational data structure for the exponential growth of kernels, input images, and feature maps during the forward pass. Written with composition in mind.  
- `main.cpp` – Prototype forward pass + demonstration script  

---

## Purpose

This project was created to build practical intuition for convolutional architectures and to improve my fluency in modern C++. It is not intended for production use or active development, and exists for demonstration and archival purposes.

---

## Author

**Samuel Clucas**  
BSc, Biological Sciences (First Class), Durham University  
Incoming MRes Student – Biomedical Data Science, Imperial College London  
