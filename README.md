# Intelligent-ecosystem-monitoring-and-prediction-platform

## Project Overview

This project focuses on developing an innovative edge AI weather forecasting system using the Sony Spresense development board. The system collects weather-related data from the Kaggle open-source database, pre-processes it, and then trains a random forest model based on historical weather data. The hyperparameters are adjusted to improve forecasting performance. The trained model is stored on a microSD card. Additionally, a camera module is integrated to capture real-time environmental images, processed using machine learning algorithms on the Sony Spresense platform.

## Key Features

- **Edge AI Implementation**: Utilizes the Sony Spresense development board for real-time data processing and weather prediction.
- **Data Collection and Preprocessing**: Gathers accurate and reliable weather data from Kaggle, followed by comprehensive cleaning and normalization.
- **Random Forest Model**: Employs a random forest model for training historical weather data, with hyperparameter tuning for optimal performance.
- **Real-Time Image Processing**: Integrates a camera module to capture and analyze real-time environmental images using machine learning algorithms.
- **Low Power Consumption**: Demonstrates the low power consumption and high computational power of the Sony Spresense kit, ideal for edge computing applications.

## Technical Description

### Data Collection and Preprocessing

- **Data Collection**: Weather-related data including temperature, humidity, wind speed, wind bearing, visibility, and cloud cover are collected from the Kaggle open-source database.
- **Data Cleaning**: Removes noise and outliers to ensure data accuracy and consistency.
- **Normalization**: Transforms data to the same scale for uniform processing and analysis.

### Model Training

- **Random Forest Model**: Constructs multiple decision trees to predict weather patterns, offering high accuracy and good generalization ability.
- **Hyperparameter Tuning**: Adjusts model parameters (number of trees, tree depth, minimum number of samples for split nodes) to find the optimal combination.

### Real-Time Image Processing

- **Camera Module Integration**: Captures real-time environmental images at fixed intervals.
- **Image Data Analysis**: Analyzes image data in real-time using machine learning algorithms on the Sony Spresense platform.

## Innovations

- **Edge AI Algorithms**: Deploys advanced Edge AI algorithms for efficient real-time data analysis and prediction, reducing reliance on cloud computing resources.
- **TinyML and Deep Learning**: Implements TinyML and deep learning models on the Spresense board, demonstrating its potential in various real-time data processing applications.

## Usage of the Spresense Kit

- **TinyML and Deep Learning**: Runs models on the Spresense board for real-time data processing and prediction, showcasing its computational capabilities and low-power advantage.
- **Camera Module**: Captures and analyzes environmental images using advanced image processing techniques to monitor and detect anomalies.

## How to Use

1. **Setup**: Connect the Sony Spresense board with the necessary sensor modules and camera.
2. **Data Collection**: Ensure the Kaggle dataset is preprocessed and ready for training.
3. **Model Training**: Train the random forest model with the historical weather data.
4. **Deployment**: Store the trained model on a microSD card and deploy it on the Sony Spresense board.
5. **Real-Time Monitoring**: Use the integrated camera module for capturing and analyzing environmental images in real-time.

## Acknowledgements

Special thanks to Sony for providing the Spresense development board and to Kaggle for the open-source weather data.
