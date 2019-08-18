# scholarlyimpact
TBA

# Abstract
TBA

# Data
TBA

# Methodology
TBA

# Results
#### Optimum parameters for the Neural network in Experiment 1
| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** | 10 |
| **Batch size** | 64 |
| **Loss Function** | binary cross-entropy |
| **Hidden Layers** | 1 layer with 512 neurons |
| **Optimization function** | RMS with 0.001 learning rate |
| **Activation function(s)** | SeLU for the hidden layer, <br>Softmax for the o/p layer |

#### Accuracy, Precision, Recall and F-1 for the neural network used in Experiment 1
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

#### Accuracy, Precision, Recall and F-1 for the supervised learning algorithms used in Experiment 1
| Model  | Train Accuracy | Test Accuracy| Precision | Recall |  F-1 |
|--------|:--------------:|:------------:|:---------:|:------:|:----:|
| **Random Forest** | 0.865 | 0.862 | 0.863 | 1.0 | 0.927 |
| **Decision Tree** | 0.865 | 0.863 | 0.863 | 1.0 | 0.927 |
| **Gradient Boosting** | 0.865 | 0.863 | 0.863 | 1.0 | 0.927 |
| **AdaBoost** | 0.87 | 0.866 | 0.87 | 0.993 | 0.928 |
| **BernouliNB** | 0.84 | 0.836 | 0.876 | 0.943 | 0.908 |
| **KNN** | 0.85 | 0.851 | 0.883 | 0.953 | 0.917 |

#### Optimum tuning parameters for the tree based and ensemble algorithms used in Experiment 1
| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | n_estimators: 2,<br> min_samples_split: 0.9,<br> min_samples_leaf: 0.3,<br> features: 18,<br> max_depth: 9,<br> criterion: gini-index |
| **Decision Tree** | min_samples_split: 0.5,<br> min_samples_leaf: 0.3,<br> max_features: 10,<br> max_depth: 32,<br> criterion: gini-index  |
| **Gradient Boosting** | n_estimators: 200,<br> min_samples_split: 0.6,<br> min_samples_leaf: 0.1,<br> max_features: 9,<br> max_depth: 4,<br> learning rate: 0.001  |

#### Optimum tuning parameters for the C-support vector machine algorithm used in Experiment 1
| Parameter  | Value |
|------------|:-----:|
| **Kernel** | Sigmoid |
| **Degree of the kernel** | 3 |
| **Tolerance** | 0.001 |
| **Gamma** | 0.045 |

#### Accuracy, Precision, Recall and F-1 for the C-support vector machine algorithm used in Experiment 1
| Metric  | Value |
|--------|:------:|
| **Training Accuracy** | 0.86 (+/- 0.00) |
| **Validation Accuracy** | 0.86 (+/- 0.01) |
| **Test Accuracy** | 0.861 |
| **Precision** | 0.864 |
| **Recall** | 0.997 |
| **F-1** | 0.925 |

# License
[The MIT License](https://github.com/akhilpandey95/scholarlyimpact/blob/master/LICENSE)

# Author(s)
[Akhil Pandey](https://github.com/akhilpandey95/akhilpandey95), [Hamed Alhoori](https://github.com/alhoori)
