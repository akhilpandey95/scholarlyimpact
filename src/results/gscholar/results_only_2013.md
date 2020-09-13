### 1. Classification experiment for predicting if citations exist or not

###### 1.a. Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** | 100 |
| **Batch size** | 32 |
| **Loss Function** | binary_crossentropy |
| **Hidden Layers** | 2 {512, 512} |
| **Optimization function** | RMS with 0.001 learning rate |
| **Activation function(s)** | selu for hidden layers, <br> softmax for output layer |

###### 1.b Accuracy, Precision, Recall and F-1 for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training Loss** | 0.8831605557406157 |
| **Training Accuracy** | 0.8631895 |
| **Test Loss** | 0.832354579723951 |
| **Test Accuracy** | 0.872803 |
| **Precision** | 0.872803 |
| **Recall** | 0.872803 |
| **F-1** | 0.8728029727935791 |

###### 1.c Accuracy, Precision, Recall and F-1 for the supervised learning algorithms

| Model  | Train Accuracy | Test Accuracy| Precision | Recall |  F-1 |
|--------|:--------------:|:------------:|:---------:|:------:|:----:|
| **Random Forest** | 0.862 | 0.864 | 0.864 | 1.0 | 0.927 |
| **Decision Tree** | 0.862 | 0.864 | 0.864 | 1.0 | 0.927 |
| **Gradient Boosting** | 0.862 | 0.864 | 0.864 | 1.0 | 0.927 |
| **AdaBoost** | 0.866 | 0.866 | 0.878 | 0.982 | 0.927 |
| **BernouliNB** | 0.837 | 0.840 | 0.876 | 0.950 | 0.911 |
| **KNN** | 0.874 | 0.845 | 0.878 | 0.953 | 0.914 |

###### 1.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | random_state: 1578, <br> n_estimators: 4,<br> min_samples_split: 0.4,<br> min_samples_leaf: 0.2,<br> max_features: 10,<br> max_depth: 4.0,<br> criterion: gini-index |
| **Decision Tree** | random_state: 2497, <br> min_samples_split: 0.8,<br> min_samples_leaf: 0.1,<br> max_features: 17,<br> max_depth: 32,<br> criterion: gini-index  |
| **Gradient Boosting** | random_state: 3833, <br> n_estimators: 64,<br> min_samples_split: 0.8,<br> min_samples_leaf: 0.1,<br> max_features: 6,<br> max_depth: 21,<br> learning rate: 0.01  |

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
| **Validation Accuracy** | 0.86 (+/- 0.02) |
| **Train Accuracy**| 0.859 |
| **Test Accuracy** | 0.862 |
| **Precision** | 0.864 |
| **Recall** | 0.998 |
| **F-1** | 0.926 |

### 2. Classification experiment for predicting if citations are more than median or not

###### 2.a Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** | 100 |
| **Batch size** | 32 |
| **Loss Function** | binary_crossentropy |
| **Hidden Layers** | 3 {64, 128, 64} |
| **Optimization function** | RMS with 0.001 learning  |
| **Activation function(s)** | elu for hidden layers <br> softmax for output layer |

###### 2.b Accuracy, Precision, Recall and F-1 for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training Loss** | 0.46954695811309716 |
| **Training Accuracy** | 0.79808027 |
| **Test Loss** | 0.5174921028708882 |
| **Test Accuracy** | 0.780296 |
| **Precision** | 0.780296 |
| **Recall** | 0.780296 |
| **F-1** | 0.7802960276603699 |

###### 2.c Accuracy, Precision, Recall and F-1 for the supervised learning algorithms

| Model  | Train Accuracy | Test Accuracy| Precision | Recall |  F-1 |
|--------|:--------------:|:------------:|:---------:|:------:|:----:|
| **Random Forest** | 0.775 | 0.771 | 0.757 | 0.783 | 0.770 |
| **Decision Tree** | 0.766 | 0.750 | 0.813 | 0.636 | 0.713 |
| **Gradient Boosting** | 0.805 | 0.800 | 0.820 | 0.757 | 0.787 |
| **AdaBoost** | 0.804 | 0.794 | 0.812 | 0.753 | 0.782 |
| **BernouliNB** | 0.674 | 0.673 | 0.730 | 0.525 | 0.611 |
| **KNN** | 0.784 | 0.689 | 0.726 | 0.584 | 0.647 |

###### 2.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | random_state: 3898, <br> n_estimators: 200,<br> min_samples_split: 0.3,<br> min_samples_leaf: 0.1,<br> features: 9,<br> max_depth: 18,<br> criterion: entropy |
| **Decision Tree** | random_state: 3916, <br> min_samples_split: 0.7,<br> min_samples_leaf: 0.1,<br> max_features: 8,<br> max_depth: 1,<br> criterion: entropy |
| **Gradient Boosting** | random_state: 4038, <br> n_estimators: 200,<br> min_samples_split: 0.1,<br> min_samples_leaf: 0.1,<br> max_features: 6,<br> max_depth: 28,<br> learning rate: 0.05 |

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
| **Validation Accuracy** | 0.59 (+/- 0.03) |
| **Train Accuracy** | 0.949 |
| **Test Accuracy** | 0.590 |
| **Precision** | 0.771 |
| **Recall** | 0.229 |
| **F-1** | 0.353 |

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
| **Training MSE** | 1.1950798 |
| **Test MSE** | 1.5005224 |
| **R-squared** | 0.44818878 |

###### 3.c MSE and R-squared for the supervised learning algorithms

| Model  | Train MSE | Test MSE| R-squared |
|--------|:---------:|:-------:|:---------:|
| **Random Forest** | 1.485 | 1.679 | 0.399 |
| **Decision Tree** | 1.456 | 1.673 | 0.401 |
| **Linear** | 1.653 | 2.009 | 0.281 |

###### 3.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | random_state: 416, <br> n_estimators: 64,<br> min_samples_split: 0.1,<br> min_samples_leaf: 0.1,<br> max_features: 17,<br> max_depth: 6,<br> criterion: mse |
| **Decision Tree** | random_state: 1762, <br> min_samples_split: 0.2,<br> min_samples_leaf: 0.1,<br> max_features: 17,<br> max_depth: 32,<br> criterion: mse  |
