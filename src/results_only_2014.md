### 1. Classification experiment for predicting if citations exist or not

###### 1.a. Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** | 100 |
| **Batch size** | 32 |
| **Loss Function** | binary_crossentropy |
| **Hidden Layers** | 2 {512, 512} |
| **Optimization function** | RMS with 0.001 learning  |
| **Activation function(s)** | selu for hidden layers, <br> softmax for output layer |

###### 1.b Accuracy, Precision, Recall and F-1 for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training Loss** | 2.0129842191353062 |
| **Training Accuracy** | 0.86875063 |
| **Test Loss** | 2.0243058100990625 |
| **Test Accuracy** | 0.8680124 |
| **Precision** | 0.8680124 |
| **Recall** | 0.8680124 |
| **F-1** | 0.8680124282836914 |

###### 1.c Accuracy, Precision, Recall and F-1 for the supervised learning algorithms

| Model  | Train Accuracy | Test Accuracy| Precision | Recall |  F-1 |
|--------|:--------------:|:------------:|:---------:|:------:|:----:|
| **Random Forest** | 0.864 | 0.860 | 0.860 | 1.0 | 0.925 |
| **Decision Tree** | 0.864 | 0.860 | 0.860 | 1.0 | 0.925 |
| **Gradient Boosting** | 0.872 | 0.877 | 0.882 | 0.992 | 0.934 |
| **AdaBoost** | 0.869 | 0.875 | 0.880 | 0.992 | 0.933 |
| **BernouliNB** | 0.850 | 0.861 | 0.887 | 0.963 | 0.924 |
| **KNN** | 0.879 | 0.856 | 0.887 | 0.957 | 0.921 |

###### 1.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | random_state: 1973, <br> n_estimators: 8,<br> min_samples_split: 0.3,<br> min_samples_leaf: 0.5,<br> max_features: 15,<br> max_depth: 7.0,<br> criterion: gini-index |
| **Decision Tree** | random_state: 20, <br> min_samples_split: 0.5,<br> min_samples_leaf: 0.5,<br> max_features: 2,<br> max_depth: 1,<br> criterion: entropy  |
| **Gradient Boosting** | random_state: 4935, <br> n_estimators: 200,<br> min_samples_split: 1.0,<br> min_samples_leaf: 0.2,<br> max_features: 20,<br> max_depth: 5,<br> learning rate: 1  |

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
| **Training Accuracy** | 0.87 (+/- 0.00) |
| **Validation Accuracy** | 0.87 (+/- 0.03) |
| **Train Accuracy**| 0.867 |
| **Test Accuracy** | 0.875 |
| **Precision** | 0.875 |
| **Recall** | 1.000 |
| **F-1** | 0.933 |

### 2. Classification experiment for predicting if citations are more than median or not

###### 2.a Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** | 100 |
| **Batch size** | 32 |
| **Loss Function** | binary_crossentropy |
| **Hidden Layers** | 3 {64, 128, 64} |
| **Optimization function** | RMS with 0.001 learning rate |
| **Activation function(s)** | elu for hidden layers, <br> softmax for output layer |

###### 2.b Accuracy, Precision, Recall and F-1 for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training Loss** | 0.437495032322313 |
| **Training Accuracy** | 0.7992428 |
| **Test Loss** | 0.5105127806989302 |
| **Test Accuracy** | 0.7802795 |
| **Precision** | 0.7802795 |
| **Recall** | 0.7802795 |
| **F-1** | 0.7802795171737671 |

###### 2.c Accuracy, Precision, Recall and F-1 for the supervised learning algorithms

| Model  | Train Accuracy | Test Accuracy| Precision | Recall |  F-1 |
|--------|:--------------:|:------------:|:---------:|:------:|:----:|
| **Random Forest** | 0.776 | 0.786 | 0.836 | 0.700 | 0.762 |
| **Decision Tree** | 0.781 | 0.795 | 0.802 | 0.769 | 0.785 |
| **Gradient Boosting** | 0.801 | 0.810 | 0.820 | 0.781 | 0.800 |
| **AdaBoost** | 0.797 | 0.803 | 0.809 | 0.780 | 0.794 |
| **BernouliNB** | 0.671 | 0.682 | 0.756 | 0.514 | 0.612 |
| **KNN** | 0.781 | 0.711 | 0.748 | 0.617 | 0.676 |

###### 2.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | random_state: 4128, <br> n_estimators: 32,<br> min_samples_split: 0.4,<br> min_samples_leaf: 0.1,<br> features: 13,<br> max_depth: 15,<br> criterion: entropy |
| **Decision Tree** | random_state: 4518, <br> min_samples_split: 0.1,<br> min_samples_leaf: 0.1,<br> max_features: 14,<br> max_depth: 32,<br> criterion: entropy  |
| **Gradient Boosting** | random_state: 1412, <br> n_estimators: 200,<br> min_samples_split: 0.8,<br> min_samples_leaf: 0.1,<br> max_features: 19,<br> max_depth: 6,<br> learning rate: 0.25  |

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
| **Training Accuracy** | 0.99 (+/- 0.00) |
| **Validation Accuracy** | 0.60 (+/- 0.02) |
| **Train Accuracy** | 0.948 |
| **Test Accuracy** | 0.591 |
| **Precision** | 0.725 |
| **Recall** | 0.262 |
| **F-1** | 0.385 |

### 3. Regression experiment for predicting log(1+citations)

###### 3.a Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** | 500 |
| **Batch size** | 32 |
| **Loss Function** | mean_squared_error |
| **Hidden Layers** | 7 {32, 64, 64, 128, 64, 64, 32} |
| **Optimization function** | RMS with 0.001 learning rate |
| **Activation function(s)** | rELU for hidden layers, <br> linear activation for output layer |

###### 3.b MSE, MAE and R-squared for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training MSE** | 1.1778095 |
| **Test MSE** | 1.4398558 |
| **R-squared** | 0.44590104 |

###### 3.c MSE and R-squared for the supervised learning algorithms

| Model  | Train MSE | Test MSE| R-squared |
|--------|:---------:|:-------:|:---------:|
| **Random Forest** | 1.518 | 1.547 | 0.425 |
| **Decision Tree** | 1.501 | 1.534 | 0.430 |
| **Linear** | 1.681 | 1.650 | 0.387 |

###### 3.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | random_state: 3142, <br> n_estimators: 8,<br> min_samples_split: 0.2,<br> min_samples_leaf: 0.1,<br> max_features: 16,<br> max_depth: 9,<br> criterion: mae |
| **Decision Tree** | random_state: 2686, <br> min_samples_split: 0.2,<br> min_samples_leaf: 0.1,<br> max_features: 8,<br> max_depth: 32,<br> criterion: friedman_mse  |
