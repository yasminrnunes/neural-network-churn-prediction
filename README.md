# neural-network-churn-prediction
## Description

This project implements a neural network model to predict customer churn in an Iranian telecommunications company. The model uses a feed-forward fully connected neural network architecture to analyze customer behavior patterns and predict the likelihood of customers leaving the service.

### Key Features
- Implements three different neural network architectures with varying hyperparameters
- Uses TensorFlow/Keras for model development
- Includes comprehensive data preprocessing and feature engineering
- Achieves high accuracy in churn prediction (96.14% on test data)

### Dataset
The project utilizes the [Iranian Churn Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset) from the UCI Machine Learning Repository, which contains:
- 3,150 customer records
- 13 features including call failures, SMS frequency, complaints, subscription length, and more
- Binary classification target (churn/no churn)
- Data collected over a 12-month period

### Problem Statement
Customer churn is a critical metric for telecommunications companies, representing the rate at which customers stop using their services. Predicting churn allows companies to:
- Identify at-risk customers
- Implement targeted retention strategies
- Reduce customer acquisition costs
- Improve customer satisfaction and loyalty

This project aims to provide an accurate prediction model that can help telecom companies proactively address customer churn.



## Project Structure

The project is organized into several Python modules, each responsible for specific aspects of the machine learning pipeline:

### Main Components

- `__Main__.py`
  - Main execution file
  - Orchestrates the entire workflow
  - Imports and runs all model variations
  - Handles dataset loading and model evaluation

### Data Processing
- `processing_data.py`
  - Handles data preprocessing
  - Manages missing values
  - Applies necessary transformations to numerical and categorical data
  - **Class Imbalance Handling**:
    - Original dataset showed significant imbalance between churn and non-churn classes
    - Applied data augmentation technique by adding 2,100 synthetic records with churn=1
    - This balanced the dataset and improved model performance
  - Splits data into training and test sets

### Model Implementations

#### Model 1: Grid Search
- `alternative1.py`
  - Implements hyperparameter grid search
  - Tests different combinations of parameters

#### Model 2: Best Parameters
- `alternative2.py`
  - Implements the best model from grid search
  - Uses optimal hyperparameters

#### Model 3: Learning Rate Optimization
- `alternative3.py`
  - Implements adaptive learning rate
  - Uses exponential decay
  - Includes callback mechanisms
  - Saves the best model checkpoint

### Model Validation
- `alternative3_validation.py`
  - Handles model validation
  - Evaluates model performance on training data
  - Performs final testing on test dataset
  - Generates performance metrics

### Saved Models
- `alternative3.keras`
  - Saved model file
  - Contains the best performing model architecture and weights

## Methodology

### Neural Network Architecture

#### Base Architecture
- **Input Layer**: 30 neurons (matching the number of features after preprocessing)
- **Hidden Layers**: 
  - First hidden layer: 96 neurons
  - Activation function: ReLU (Rectified Linear Unit)
- **Output Layer**: 
  - 1 neuron (binary classification)
  - Activation function: Sigmoid
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy

### Model Development Process

#### Model 1: Hyperparameter Optimization
- **Purpose**: Find optimal hyperparameters through grid search
- **Tested Parameters**:
  - Neurons: [32, 64, 96]
  - Optimizers: 
    - SGD (Stochastic Gradient Descent)
    - RMSprop
    - Adam
  - Epochs: [20, 40]
  - Batch sizes: [64, 128, 256]
- **Best Configuration**:
  - Neurons: 96
  - Optimizer: Adam
  - Epochs: 40
  - Batch size: 64
  - Accuracy: 94.24%

#### Model 2: Best Parameters Implementation
- **Purpose**: Implement and evaluate the best configuration from Model 1
- **Performance Metrics**:
  - Training accuracy: 95.68%
  - Training loss: 0.125
  - Validation accuracy: 94.40%
  - Validation loss: 0.157
- **Observations**:
  - Model showed potential for further improvement
  - Training and validation curves showed slight separation
  - Higher prediction errors in non-churn cases

#### Model 3: Learning Rate Optimization
- **Purpose**: Improve model performance through adaptive learning rate
- **Key Features**:
  - Exponential decay learning rate
  - Callback implementation for model checkpointing
  - Early stopping to prevent overfitting
- **Performance Metrics**:
  - Training accuracy: 96.58%
  - Training loss: 0.105
  - Validation accuracy: 96.55%
  - Validation loss: 0.127
  - Test accuracy: 96.14%
  - Test loss: 0.114
- **Best Epoch Performance**:
  - Achieved at epoch 39
  - Training accuracy: 96.92%
  - Training loss: 0.105
  - This was the selected model checkpoint for final evaluation

### Training Methodology

1. **Data Preprocessing**:
   - Feature scaling
   - Handling categorical variables
   - Class balancing
   - Train-test split

2. **Model Training**:
   - Batch-wise training
   - Validation split during training
   - Model checkpointing
   - Learning rate adaptation

3. **Evaluation**:
   - Confusion matrix analysis
   - Accuracy and loss metrics
   - Validation on separate test set

### Model Selection Criteria

The final model (Model 3) was selected based on:
- Highest overall accuracy
- Balanced performance between training and validation
- Stable learning curves
- Good generalization on test data
- Low standard deviation in predictions

### Performance Analysis

- **Training-Validation Gap**: Minimal, indicating good generalization
- **Error Distribution**: Balanced between churn and non-churn predictions
- **Model Stability**: Consistent performance across different data splits
- **Final Test Performance**: 96.14% accuracy, demonstrating robust predictive power


## Requirements

### Dependencies
- Python 3.x
- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- ucimlrepo


## Usage

### Running the Model
```python
# Import the main module
import __Main__

# The script will automatically:
# 1. Load and preprocess the data
# 2. Train the three model variations
# 3. Select and save the best model
# 4. Evaluate the final model on test data
```

### Making Predictions
```python
# Load the saved model
model = keras.models.load_model('alternative3.keras')

# Make predictions
predictions = model.predict(new_data)
```

## Results and Analysis

### Model Performance Comparison
| Model | Training Accuracy | Validation Accuracy | Test Accuracy |
|-------|------------------|-------------------|---------------|
| Model 1 | 94.24% | - | - |
| Model 2 | 95.68% | 94.40% | - |
| Model 3 | 96.92% | 96.55% | 96.14% |

### Key Findings
- Model 3 achieved the best performance with 96.14% accuracy on test data
- The adaptive learning rate and early stopping helped prevent overfitting
- The balanced dataset through data augmentation improved model performance
- The model shows good generalization with minimal gap between training and validation metrics

## Future Improvements
- Experiment with different neural network architectures
- Try other techniques for handling class imbalance
- Implement cross-validation for more robust evaluation
- Add feature importance analysis
- Explore ensemble methods

## References
- [Iranian Churn Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)
- TensorFlow Documentation
- UCI Machine Learning Repository


## Author
Yasmin Nunes
