# A1 - Neural Network with Back-Propagation Project - SEBASTIAN BUZDUGAN

## Overview
This project focuses on implementing a Neural Network using the Back-Propagation algorithm (BP) from scratch. The aim is to understand the core principles of neural computation and to compare the performance of this custom implementation with a pre-existing Back-Propagation algorithm (BP-F) and Multiple Linear Regression (MLR-F) on three different datasets.

## Project Structure
The project is divided into three main parts:

### Part 1: Selecting and Analyzing the Datasets
- **Datasets:**
  - `A1-turbine.txt` - Consists of 5 features, with the first 4 as input variables and the last one as the value to predict.
  - `A1-synthetic.txt` - Comprises 10 features, with the first 9 as input variables and the last one as the value to predict.
  - A dataset sourced from the Internet, meeting specific criteria outlined in the assignment.
- **Data Processing:**
  - Explanation of data normalization techniques applied.
  - Documentation of preprocessing methods for the third dataset.

### Part 2: Implementation of Back-Propagation (BP)
- **Implementation Details:**
  - Implementation using Python (version >= 3.6).
  - Detailed adherence to provided algorithm and equations in document [G].
  - Use of specific array structures for network components (weights, activations, etc.).
  - Code structure to handle arbitrary multilayer networks.
  - Division of input dataset into training and validation sets.
- **MyNeuralNetwork Class:**
  - Constructor parameters (layers, units, epochs, learning rate, momentum, activation function, validation set percentage).
  - Public functions: `fit`, `predict`, and `loss_epochs`.

### Part 3: Obtaining and Comparing Predictions
- **Comparison Strategy:**
  - Using Jupyter Notebooks for coding and analysis.
  - Utilization of Mean Absolute Percentage Error (MAPE) for assessing prediction quality.
  - Visualization of results with scatter plots.
- **Parameter Comparison and Selection:**
  - Exploration and documentation of optimal network parameters for each dataset.
  - Presentation of prediction quality results in tabular form.
  - Representative scatter plots and error evolution plots.
- **Model Result Comparison:**
  - Comparison of custom BP, BP-F, and MLR-F models.
  - Documentation of parameters used in the BP-F and MLR-F models.
  - Comparative analysis of prediction quality (MAPE) for all three models.
  - Scatter plots of predicted vs. real values for each model.


