# scholarlyimpact
TBA

# Abstract
TBA

# Data
TBA

# Methodology
TBA

# Results

### 1. Classification experiment for predicting if citations exist or not

###### 1.a. Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** | 10 |
| **Batch size** | 64 |
| **Loss Function** | binary cross-entropy |
| **Hidden Layers** | 1 layer with 512 neurons |
| **Optimization function** | RMS with 0.001 learning rate |
| **Activation function(s)** | SeLU for the hidden layer, <br>Softmax for the o/p layer |

###### 1.b Accuracy, Precision, Recall and F-1 for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training Loss** | 2.07495 |
| **Training Accuracy** | 0.8646 |
| **Validation Loss** | 2.0891 |
| **Validation Accuracy** | 0.86368 |
| **Test Accuracy** | 0.8655 |
| **Precision** | 0.866 |
| **Recall** | 1.0 |
| **F-1** | 0.9279 |

###### 1.c Accuracy, Precision, Recall and F-1 for the supervised learning algorithms

| Model  | Train Accuracy | Test Accuracy| Precision | Recall |  F-1 |
|--------|:--------------:|:------------:|:---------:|:------:|:----:|
| **Random Forest** | 0.865 | 0.862 | 0.863 | 1.0 | 0.927 |
| **Decision Tree** | 0.865 | 0.863 | 0.863 | 1.0 | 0.927 |
| **Gradient Boosting** | 0.865 | 0.863 | 0.863 | 1.0 | 0.927 |
| **AdaBoost** | 0.87 | 0.866 | 0.87 | 0.993 | 0.928 |
| **BernouliNB** | 0.84 | 0.836 | 0.876 | 0.943 | 0.908 |
| **KNN** | 0.85 | 0.851 | 0.883 | 0.953 | 0.917 |

###### 1.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | n_estimators: 2,<br> min_samples_split: 0.9,<br> min_samples_leaf: 0.3,<br> features: 18,<br> max_depth: 9,<br> criterion: gini-index |
| **Decision Tree** | min_samples_split: 0.5,<br> min_samples_leaf: 0.3,<br> max_features: 10,<br> max_depth: 32,<br> criterion: gini-index  |
| **Gradient Boosting** | n_estimators: 200,<br> min_samples_split: 0.6,<br> min_samples_leaf: 0.1,<br> max_features: 9,<br> max_depth: 4,<br> learning rate: 0.001  |

###### 1.e Optimum tuning parameters for the C-support vector machine algorithm

| Parameter  | Value |
|------------|:-----:|
| **Kernel** | Sigmoid |
| **Degree of the kernel** | 3 |
| **Tolerance** | 0.001 |
| **Gamma** | 0.045 |

###### 1.f Accuracy, Precision, Recall and F-1 for the C-support vector machine algorithm

| Metric  | Value |
|--------|:------:|
| **Training Accuracy** | 0.86 (+/- 0.00) |
| **Validation Accuracy** | 0.86 (+/- 0.01) |
| **Test Accuracy** | 0.861 |
| **Precision** | 0.864 |
| **Recall** | 0.997 |
| **F-1** | 0.925 |

### 2. Classification experiment for predicting if citations are more than median or not

###### 2.a Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** | 10 |
| **Batch size** | 64 |
| **Loss Function** | binary cross-entropy |
| **Hidden Layers** | 3 layers 64, 128, 64 neurons for respective layers |
| **Optimization function** | RMS with 0.001 learning rate |
| **Activation function(s)** | SeLU for the second hidden layer, <br>Sigmoid for remaining layers |

###### 2.b Accuracy, Precision, Recall and F-1 for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training Loss** | 0.4710 |
| **Training Accuracy** | 0.78006 |
| **Validation Loss** | 0.4726 |
| **Validation Accuracy** | 0.7797 |
| **Test Accuracy** | 0.7794 |
| **Precision** | 0.81918 |
| **Recall** | 0.69692 |
| **F-1** | 0.75312 |

###### 2.c Accuracy, Precision, Recall and F-1 for the supervised learning algorithms

| Model  | Train Accuracy | Test Accuracy| Precision | Recall |  F-1 |
|--------|:--------------:|:------------:|:---------:|:------:|:----:|
| **Random Forest** | 0.778 | 0.784 | 0.799 | 0.737 | 0.767 |
| **Decision Tree** | 0.773 | 0.768 | 0.725 | 0.834 | 0.776 |
| **Gradient Boosting** | 0.802 | 0.80 | 0.810 | 0.767 | 0.788 |
| **AdaBoost** | 0.80 | 0.797 | 0.806 | 0.760 | 0.782 |
| **BernouliNB** | 0.67 | 0.674 | 0.742 | 0.495 | 0.594 |
| **KNN** | 0.75 | 0.752 | 0.777 | 0.680 | 0.725 |

###### 2.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | n_estimators: 100,<br> min_samples_split: 0.1,<br> min_samples_leaf: 0.1,<br> features: 3,<br> max_depth: 30,<br> criterion: entropy |
| **Decision Tree** | min_samples_split: 0.4,<br> min_samples_leaf: 0.1,<br> max_features: 15,<br> max_depth: 32,<br> criterion: entropy  |
| **Gradient Boosting** | n_estimators: 200,<br> min_samples_split: 0.1,<br> min_samples_leaf: 0.1,<br> max_features: 12,<br> max_depth: 20,<br> learning rate: 0.005  |

###### 2.e Optimum tuning parameters for the C-support vector machine algorithm

| Parameter  | Value |
|------------|:-----:|
| **Kernel** | Sigmoid |
| **Degree of the kernel** | 3 |
| **Tolerance** | 0.001 |
| **Gamma** | 0.045 |

###### 2.f Accuracy, Precision, Recall and F-1 for the C-support vector machine algorithm

| Metric  | Value |
|--------|:------:|
| **Training Accuracy** | 0.52 (+/- 0.00) |
| **Validation Accuracy** | 0.52 (+/- 0.01) |
| **Test Accuracy** | 0.519 |
| **Precision** | 0.478 |
| **Recall** | 0.007 |
| **F-1** | 0.014 |

# License
[The MIT License](https://github.com/akhilpandey95/scholarlyimpact/blob/master/LICENSE)

# Author(s)
[Akhil Pandey](https://github.com/akhilpandey95/akhilpandey95), [Hamed Alhoori](https://github.com/alhoori)
