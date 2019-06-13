
# Using a neural network model for predicting the citations of a scholarly paper

```python
# the following data columns will be considered as features
data_columns = ['mendeley', 'citeulike', 'News', 'Blogs',
                'Reddit', 'Twitter', 'Facebook',
                'GooglePlus', 'PeerReviews','Wikipedia',
                'TotalPlatforms', 'SincePublication','PlatformWithMaxMentions',
                'Countries', 'MaxFollowers', 'Retweets','Profession',
                'AcademicStatus', 'PostLength', 'HashTags', 'Mentions',
                'AuthorCount']

# set the X column
X = data.as_matrix(columns = data_columns)

# set the target variable
Y = data.target_exp_3

# train and test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
```
### Run 1 with arch: [22, 64, 64, 1]

```python
regressor = Sequential()

regressor.add(Dense(64, activation='relu', input_dim=22))

regressor.add(Dense(64, activation='relu'))

regressor.add(Dense(1))
```

    INFO:plaidml:Opening device "metal_amd_radeon_pro_460.0"



```python
regressor.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 64)                1472      
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 5,697
    Trainable params: 5,697
    Non-trainable params: 0
    _________________________________________________________________



```python
# use the rmsprop optimizer
rms = keras.optimizers.RMSprop(lr=0.001)

# prepare for early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0,
                                         patience=40, verbose=0, mode='auto',
                                         baseline=None, restore_best_weights=False)

# compile the model
regressor.compile(optimizer=rms, loss='mean_squared_error',
                   metrics =['mean_absolute_error', 'mean_squared_error'])
```


```python
regressor.fit(X_train, Y_train, epochs=100,
               validation_split=0.2, callbacks=[early_stopping])
```

    Train on 73216 samples, validate on 18305 samples

    Epoch 1/100
    73216/73216 [==============================] - 16s 220us/step - loss: 2455946.6292 - mean_absolute_error: 123.7999 - mean_squared_error: 2455946.6292 - val_loss: 2924590.0726 - val_mean_absolute_error: 227.9805 - val_mean_squared_error: 2924590.0726
    Epoch 10/100
    73216/73216 [==============================] - 16s 215us/step - loss: 5640.0851 - mean_absolute_error: 6.0561 - mean_squared_error: 5640.0851 - val_loss: 3.2153 - val_mean_absolute_error: 1.1301 - val_mean_squared_error: 3.2153
    Epoch 25/100
    73216/73216 [==============================] - 16s 224us/step - loss: 16.8282 - mean_absolute_error: 1.1826 - mean_squared_error: 16.8282 - val_loss: 2.8397 - val_mean_absolute_error: 0.9878 - val_mean_squared_error: 2.8397
    Epoch 50/100
    73216/73216 [==============================] - 15s 210us/step - loss: 2.4281 - mean_absolute_error: 0.9240 - mean_squared_error: 2.4281 - val_loss: 1.4658 - val_mean_absolute_error: 0.9040 - val_mean_squared_error: 1.4658
    Epoch 75/100
    73216/73216 [==============================] - 15s 211us/step - loss: 1.6651 - mean_absolute_error: 0.9064 - mean_squared_error: 1.6651 - val_loss: 1.4983 - val_mean_absolute_error: 0.9258 - val_mean_squared_error: 1.4983
    Epoch 100/100
    73216/73216 [==============================] - 16s 225us/step - loss: 2.8220 - mean_absolute_error: 0.9104 - mean_squared_error: 2.8220 - val_loss: 1.7256 - val_mean_absolute_error: 0.8737 - val_mean_squared_error: 1.7256

    91521/91521 [==============================] - 10s 109us/step
    Training Loss: 1.6389000566827658
    Training MAE: 0.8746740578409375
    Training MSE: 1.6389000566827658

    39224/39224 [==============================] - 4s 105us/step
    Test Loss: 1.3873362067814095
    Test MAE: 0.8774775451187112
    Test MSE: 1.3873362067814095

    R-squared value: 0.48304519060143325

### Run 2 with arch: [22, 32, 64, 64, 32, 1]


```python
regressor = Sequential()

regressor.add(Dense(32, activation='relu', input_dim=22))

regressor.add(Dense(64, activation='relu'))

regressor.add(Dense(64, activation='relu'))

regressor.add(Dense(32, activation='relu'))

regressor.add(Dense(1))
```

    INFO:plaidml:Opening device "metal_amd_radeon_pro_460.0"



```python
regressor.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 32)                736       
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                2112      
    _________________________________________________________________
    dense_3 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_4 (Dense)              (None, 32)                2080      
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 9,121
    Trainable params: 9,121
    Non-trainable params: 0
    _________________________________________________________________



```python
# use the rmsprop optimizer
rms = keras.optimizers.RMSprop(lr=0.001)
sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
nesterov = keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=True)

# prepare for early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='mean_squared_error', min_delta=0,
                                         patience=40, verbose=0, mode='auto',
                                         baseline=None, restore_best_weights=False)

# compile the model
regressor.compile(optimizer=rms, loss='mean_squared_error',
                   metrics =['mean_absolute_error', 'mean_squared_error'])
```


```python
regressor.fit(X_train, Y_train, epochs=50,
               validation_split=0.2, callbacks=[early_stopping])
```

    Train on 73216 samples, validate on 18305 samples
    Epoch 1/50
    73216/73216 [==============================] - 21s 290us/step - loss: 365527.9862 - mean_absolute_error: 38.2034 - mean_squared_error: 365527.9862 - val_loss: 1949.8650 - val_mean_absolute_error: 7.0868 - val_mean_squared_error: 1949.8650
    Epoch 10/50
    73216/73216 [==============================] - 21s 282us/step - loss: 1.8283 - mean_absolute_error: 0.9288 - mean_squared_error: 1.8283 - val_loss: 1.4892 - val_mean_absolute_error: 0.9479 - val_mean_squared_error: 1.4892
    Epoch 20/50
    73216/73216 [==============================] - 18s 247us/step - loss: 1.5092 - mean_absolute_error: 0.8942 - mean_squared_error: 1.5092 - val_loss: 1.3909 - val_mean_absolute_error: 0.8843 - val_mean_squared_error: 1.3909
    Epoch 30/50
    73216/73216 [==============================] - 19s 263us/step - loss: 1.6611 - mean_absolute_error: 0.8882 - mean_squared_error: 1.6611 - val_loss: 1.3951 - val_mean_absolute_error: 0.8766 - val_mean_squared_error: 1.3951
    Epoch 40/50
    73216/73216 [==============================] - 18s 246us/step - loss: 1.3925 - mean_absolute_error: 0.8840 - mean_squared_error: 1.3925 - val_loss: 1.3273 - val_mean_absolute_error: 0.8623 - val_mean_squared_error: 1.3273
    Epoch 50/50
    73216/73216 [==============================] - 17s 234us/step - loss: 1.4372 - mean_absolute_error: 0.8774 - mean_squared_error: 1.4372 - val_loss: 1.4222 - val_mean_absolute_error: 0.9054 - val_mean_squared_error: 1.4222


    91521/91521 [==============================] - 10s 113us/step
    Training Loss: 1.412802493789042
    Training MAE: 0.8983146420629279
    Training MSE: 1.412802493789042

    39224/39224 [==============================] - 5s 122us/step
    Test Loss: 1.4794692387027495
    Test MAE: 0.9028650939452413
    Test MSE: 1.4794692387027495

    R-squared value: 0.4580961781182158
