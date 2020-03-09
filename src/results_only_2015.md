### 1. Classification experiment for predicting if citations exist or not

###### 1.a. Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** |  |
| **Batch size** |  |
| **Loss Function** |  |
| **Hidden Layers** |  |
| **Optimization function** | RMS with 0.001 learning  |
| **Activation function(s)** |  |

###### 1.b Accuracy, Precision, Recall and F-1 for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training Loss** |  |
| **Training Accuracy** |  |
| **Validation Loss** |  |
| **Validation Accuracy** |  |
| **Test Accuracy** |  |
| **Precision** |  |
| **Recall** |  |
| **F-1** | |

###### 1.c Accuracy, Precision, Recall and F-1 for the supervised learning algorithms

| Model  | Train Accuracy | Test Accuracy| Precision | Recall |  F-1 |
|--------|:--------------:|:------------:|:---------:|:------:|:----:|
| **Random Forest** | 0.867 | 0.875 | 0.875 | 1.0 | 0.933 |
| **Decision Tree** | 0.867 | 0.875 | 0.875 | 1.0 | 0.933 |
| **Gradient Boosting** | 0.864 | 0.860 | 0.860 | 1.0 | 0.925 |
| **AdaBoost** | 0.869 | 0.864 | 0.872 | 0.987 | 0.926 |
| **BernouliNB** | 0.845 | 0.844 | 0.870 | 0.962 | 0.914 |
| **KNN** | 0.878 | 0.835 | 0.873 | 0.945 | 0.908 |

###### 1.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | random_state: 1748, <br> n_estimators: 32,<br> min_samples_split: 0.2,<br> min_samples_leaf: 0.2,<br> max_features: 4,<br> max_depth: 7.0,<br> criterion: entropy |
| **Decision Tree** | random_state: 1833, <br> min_samples_split: 0.5,<br> min_samples_leaf: 0.2,<br> max_features: 5,<br> max_depth: 1,<br> criterion: entropy  |
| **Gradient Boosting** | random_state: 4114, <br> n_estimators: 100,<br> min_samples_split: 0.5,<br> min_samples_leaf: 0.1,<br> max_features: 11,<br> max_depth: 32,<br> learning rate: 0.25  |

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
| **Train Accuracy**| 0.861 |
| **Test Accuracy** | 0.854 |
| **Precision** | 0.859 |
| **Recall** | 0.993 |
| **F-1** | 0.921 |

### 2. Classification experiment for predicting if citations are more than median or not

###### 2.a Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** |  |
| **Batch size** |  |
| **Loss Function** |  |
| **Hidden Layers** |  |
| **Optimization function** | RMS with 0.001 learning rate |
| **Activation function(s)** |  |

###### 2.b Accuracy, Precision, Recall and F-1 for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training Loss** |  |
| **Training Accuracy** |  |
| **Validation Loss** |  |
| **Validation Accuracy** |  |
| **Test Accuracy** |  |
| **Precision** |  |
| **Recall** |  |
| **F-1** |  |

###### 2.c Accuracy, Precision, Recall and F-1 for the supervised learning algorithms

| Model  | Train Accuracy | Test Accuracy| Precision | Recall |  F-1 |
|--------|:--------------:|:------------:|:---------:|:------:|:----:|
| **Random Forest** | 0.766 | 0.771 | 0.746 | 0.814 | 0.778 |
| **Decision Tree** | 0.776 | 0.784 | 0.773 | 0.797 | 0.785 |
| **Gradient Boosting** | 0.787 | 0.795 | 0.813 | 0.762 | 0.786 |
| **AdaBoost** | 0.791 | 0.797 | 0.812 | 0.768 | 0.789 |
| **BernouliNB** | 0.657 | 0.660 | 0.737 | 0.486 | 0.586 |
| **KNN** | 0.781 | 0.707 | 0.743 | 0.625 | 0.679 |

###### 2.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | random_state: 2105, <br> n_estimators: 200,<br> min_samples_split: 0.1,<br> min_samples_leaf: 0.1,<br> features: 21,<br> max_depth: 16,<br> criterion: gini |
| **Decision Tree** | random_state: 2246, <br> min_samples_split: 0.1,<br> min_samples_leaf: 0.1,<br> max_features: 21,<br> max_depth: 32,<br> criterion: gini  |
| **Gradient Boosting** | random_state: 243, <br> n_estimators: 100,<br> min_samples_split: 0.8,<br> min_samples_leaf: 0.2,<br> max_features: 12,<br> max_depth: 13,<br> learning rate: 0.1  |

###### 2.e Optimum tuning parameters for the C-support vector machine algorithm

| Parameter  | Value |
|------------|:-----:|
| **Kernel** | RBF |
| **Degree of the kernel** | 3 |
| **Tolerance** | 0.001 |
| **Gamma** | 0.045 |

###### 2.f Accuracy, Precision, Recall and F-1 for the C-support vector machine algorithm

| Metric  | Value |
|--------|:------:|
| **Training Accuracy** | 0.98 (+/- 0.00) |
| **Validation Accuracy** | 0.63 (+/- 0.01) |
| **Train Accuracy** | 0.947 |
| **Test Accuracy** | 0.634 |
| **Precision** | 0.587 |
| **Recall** | 0.884 |
| **F-1** | 0.705 |

### 3. Regression experiment for predicting log(1+citations)

###### 3.a Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** |  |
| **Batch size** |  |
| **Loss Function** |  |
| **Hidden Layers** |  |
| **Optimization function** | RMS with 0.001 learning rate |
| **Activation function(s)** |  |

###### 3.b MSE, MAE and R-squared for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training MSE** |  |
| **Training MAE** |  |
| **Test MSE** |  |
| **Test MAE** |  |
| **Test MSE** |  |

###### 3.c MSE and R-squared for the supervised learning algorithms

| Model  | Train MSE | Test MSE| R-squared |
|--------|:---------:|:-------:|:---------:|
| **Random Forest** | 1.528 | 1.486 | 0.431 |
| **Decision Tree** | 1.513 | 1.471 | 0.437 |
| **Linear** | 1.668 | 1.724 | 0.340 |

###### 3.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | random_state: 3397, <br> n_estimators: 200,<br> min_samples_split: 0.1,<br> min_samples_leaf: 0.1,<br> features: 14,<br> max_depth: 23,<br> criterion: mae |
| **Decision Tree** | random_state: 2837, <br> min_samples_split: 0.2,<br> min_samples_leaf: 0.1,<br> max_features: 20,<br> max_depth: 32,<br> criterion: mse  |
