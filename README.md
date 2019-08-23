# scholarlyimpact

# Abstract
Identifying highly-cited scholarly literature at an early stage is a vital endeavor to the academic research community and to other stakeholders, such as technology companies and government bodies. Due to the sheer amount of research published and the growth of ever-changing interdisciplinary areas, researchers need an effective approach to identifying important scholarly studies if they are to read or even skim all the new studies published in their respective fields. The number of citations that a given research publication has accrued has been used to help researchers in this regard. However, citations take time to occur and longer to accumulate. In this article, we used Altmetrics to predict citations that a scholarly publication could receive. We built various classification and regression models and evaluated their performance. We found that tree-based models performed best in classification. We found that Mendeley readership, publication age, post length, maximum followers, and academic status were the most important factors in predicting citations.

# Data
The dataset used for the experiments comprises of social media and scholarly indicators for scientific articles. The size of the dataset is 130,745 and for all the experiments 70 percent was used for training and 30 percent for test. Furthermore, for all the neural network models 20 percent of the training data was used as validation. There are three target variables in the dataset respectively for the three experiments. These are :
  - `target_exp_1`: Binary label saying if citations exist or not.
  - `target_exp_2`: Binary label saying if existing citations are more than median number of citations or not.
  - `target_exp_3`: Discrete values for log(1 + citation).

# Methodology
The project comprises of three experiments. The first two experiments are classification problems while the third being a regression problem. A combination of approaches were used for solving the classification and regression problems. Neural networks, supervised learning algorithms and support vector machines were used for training the models on the data. The features used for all of the experiments were the same. While training the supervised learning models three algorithms had Randomized search and Grid search to obtain best hyper parameters for training the models. All of the supervised learning algorithms used 10 fold cross validation approach for training the models. The neural network models were implemented using TensorFlow. The supervised learning models and the support vector machine algorithms were implemented using scikit-learn.

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

### 3. Regression experiment for predicting log(1+citations)

###### 3.a Optimum parameters for the Neural network

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** | 500 |
| **Batch size** | 128 |
| **Loss Function** | mean squared error |
| **Hidden Layers** | 7 layers <br>32, 64, 64, 128, 64, 64, 32 neurons for respective layers |
| **Optimization function** | RMS with 0.001 learning rate |
| **Activation function(s)** | ReLU for all layers, <br> linear activation for o/p layer |

###### 3.b MSE, MAE and R-squared for the neural network

| Metric  | Value |
|---------|:-----:|
| **Training MSE** | 1.24965 |
| **Training MAE** | 0.84157 |
| **Test MSE** | 1.29756 |
| **Test MAE** | 0.85583 |
| **Test MSE** | 0.52284 |

###### 3.c MSE and R-squared for the supervised learning algorithms

| Model  | Train MSE | Test MSE| R-squared |
|--------|:---------:|:-------:|:---------:|
| **Random Forest** | 0.26 | 1.32 | 0.512 |
| **Decision Tree** | 1.647 | 1.663 | 0.389 |
| **Linear** | 1.75 | 1.758 | 0.354 |

###### 3.d Optimum tuning parameters for the tree based and ensemble algorithms

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Random Forest** | n_estimators: 16,<br> min_samples_split: 0.4,<br> min_samples_leaf: 0.2,<br> features: 16,<br> max_depth: 24,<br> criterion: mse |
| **Decision Tree** | min_samples_split: 0.4,<br> min_samples_leaf: 0.1,<br> max_features: 13,<br> max_depth: 32,<br> criterion: friedman mse  |

# License
[The MIT License](https://github.com/akhilpandey95/scholarlyimpact/blob/master/LICENSE)

# Author(s)
[Akhil Pandey](https://github.com/akhilpandey95/akhilpandey95), [Hamed Alhoori](https://github.com/alhoori)
