##Using dimensions citations for the models predicting scholarly impact

### 1. Classification experiment for predicting if citations exist or not

###### 1.a. Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** | 100 |
| **Batch size** | 32 |
| **Loss Function** | binary_crossentropy |
| **Hidden Layers** | 2 [1024, 1024] |
| **Optimization function** | RMS with 0.001 learning  |
| **Activation function(s)** | ReLU for hidden layer, <br> linear for output layer |

###### 1.b Accuracy, Precision, Recall and F-1 for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training Loss** | 2.402296781539917 |
| **Training Accuracy** | 0.843366801738739 |
| **Test Loss** | 2.211094379425049 |
| **Test Accuracy** | 0.8558335304260254 |
| **Precision** | 0.8558335304260254 |
| **Recall** | 0.8558335304260254 |
| **F-1** | 0.8558335304260254 |

###### 1.c Accuracy, Precision, Recall and F-1 for the supervised learning algorithms

| Model  | Train Accuracy | Test Accuracy| Precision | Recall |  F-1 |
|--------|:--------------:|:------------:|:---------:|:------:|:----:|
| **Random Forest** | 0.845 | 0.848 | 0.848 | 1.0 | 0.918 |
| **Decision Tree** | 0.845 | 0.848 | 0.848 | 1.0 | 0.918 |
| **Gradient Boosting** | 0.845 | 0.848 | 0.848 | 1.0 | 0.918 |
| **AdaBoost** | 0.846 | 0.848 | 0.849 | 0.999 | 0.918 |
| **BernouliNB** | 0.845 | 0.848 | 0.848 | 1.0 | 0.918 |
| **KNN** | 0.849 | 0.829 | 0.849 | 0.971 | 0.906 |

###### 1.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | random_state: 2198, <br> n_estimators: 8,<br> min_samples_split: 0.9,<br> min_samples_leaf: 0.2,<br> max_features: 14,<br> max_depth: 3.0,<br> criterion: gini |
| **Decision Tree** | random_state: 1342, <br> min_samples_split: 0.30000000000000004,<br> min_samples_leaf: 0.2,<br> max_features: 16,<br> max_depth: 32,<br> criterion: gini  |
| **Gradient Boosting** | random_state: 3859, <br> n_estimators: 100,<br> min_samples_split: 0.30000000000000004,<br> min_samples_leaf: 0.5,<br> max_features: 11,<br> max_depth: 14,<br> learning rate: 0.1  |

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
| **Train Accuracy**| 0.720 |
| **Test Accuracy** | 0.729 |
| **Precision** | 0.732 |
| **Recall** | 0.994 |
| **F-1** | 0.843 |

### 2. Classification experiment for predicting if citations are more than median or not

###### 2.a Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** | 100 |
| **Batch size** | 32 |
| **Loss Function** | binary_crossentropy |
| **Hidden Layers** | 3[64, 1024, 64] |
| **Optimization function** | RMS with 0.001 learning rate |
| **Activation function(s)** | ReLU for hidden layers, <br> Softmax for output layer |

###### 2.b Accuracy, Precision, Recall and F-1 for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training Loss** | 0.6937 |
| **Training Accuracy** | 0.5303 |
| **Validation Loss** | 0.7061 |
| **Test Accuracy** | 0.5178 |
| **Precision** | 0.5178 |
| **Recall** | 0.5178 |
| **F-1** | 0.5178 |

###### 2.c Accuracy, Precision, Recall and F-1 for the supervised learning algorithms

| Model  | Train Accuracy | Test Accuracy| Precision | Recall |  F-1 |
|--------|:--------------:|:------------:|:---------:|:------:|:----:|
| **Random Forest** | 0.939 | 0.508 | 0.483 | 0.415 | 0.446 |
| **Decision Tree** | 0.937 | 0.508 | 0.485 | 0.477 | 0.481 |
| **Gradient Boosting** | 0.550 | 0.514 | 0.474 | 0.167 | 0.247 |
| **AdaBoost** | 0.791 | 0.797 | 0.812 | 0.768 | 0.789 |
| **BernouliNB** | 0.527 | 0.526 | 0.525 | 0.072 | 0.126 |
| **KNN** | 0.675 | 0.495 | 0.470 | 0.444 | 0.457 |

###### 2.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | random_state: 789, <br> n_estimators: 100,<br> min_samples_split: 2,<br> min_samples_leaf: 1,<br> max_features: auto,<br> max_depth: None,<br> criterion: entropy |
| **Decision Tree** | random_state: 1717, <br> min_samples_split: 2,<br> min_samples_leaf: 1,<br> max_features: None,<br> max_depth: None,<br> criterion: gini  |
| **Gradient Boosting** | random_state: 1985, <br> n_estimators: 200,<br> min_samples_split: 0.3,<br> min_samples_leaf: 0.1,<br> max_features: 5,<br> max_depth: 15,<br> learning rate: 0.1  |

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
| **Train Accuracy** | 0.908 |
| **Test Accuracy** | 0.513 |
| **Precision** | 0.473 |
| **Recall** | 0.174 |
| **F-1** | 0.255 |

### 3. Regression experiment for predicting log(1+citations)

###### 3.a Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** | 500 |
| **Batch size** | 32 |
| **Loss Function** | mean_squared_error |
| **Hidden Layers** | 3[256, 512, 256] |
| **Optimization function** | RMS with 0.001 learning rate |
| **Activation function(s)** | ReLU for hidden layers, <br> linear for output layer |

###### 3.b MSE, MAE and R-squared for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training MSE** | 0.971 |
| **Test MSE** | 1.269 |
| **R-squared** | -0.248 |

###### 3.c MSE and R-squared for the supervised learning algorithms

| Model  | Train MSE | Test MSE| R-squared |
|--------|:---------:|:-------:|:---------:|
| **Random Forest** | 0.987 | 0.985 | -0.001 |
| **Decision Tree** | 0.985 | 0.984 | -0.001 |
| **Linear** | 0.985 | 0.985 | -0.001 |

###### 3.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | random_state: 2105, <br> n_estimators: 64,<br> min_samples_split: 0.1,<br> min_samples_leaf: 0.1,<br> features: 13,<br> max_depth: 12,<br> criterion: mae |
| **Decision Tree** | random_state: 2826, <br> min_samples_split: 0.1,<br> min_samples_leaf: 0.1,<br> max_features: 17,<br> max_depth: 32,<br> criterion: friedman_mse  |
