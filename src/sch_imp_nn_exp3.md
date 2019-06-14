
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

### Run 3 with arch: [22, 32, 64, 64, 64, 32, 1]
```python
regressor = Sequential()

regressor.add(Dense(32, activation='relu', input_dim=22))

regressor.add(Dense(64, activation='relu'))

regressor.add(Dense(64, activation='relu'))

regressor.add(Dense(64, activation='relu'))

regressor.add(Dense(32, activation='relu'))

regressor.add(Dense(1))
```
    /GPU:0 - NVIDIA Corporation GK210GL [Tesla K80]

```python
regressor.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_17 (Dense)             (None, 32)                736       
    _________________________________________________________________
    dense_18 (Dense)             (None, 64)                2112      
    _________________________________________________________________
    dense_19 (Dense)             (None, 64)                4160      
    _________________________________________________________________
    dense_20 (Dense)             (None, 64)                4160      
    _________________________________________________________________
    dense_21 (Dense)             (None, 32)                2080      
    _________________________________________________________________
    dense_22 (Dense)             (None, 1)                 33        
    =================================================================
    Total params: 13,281
    Trainable params: 13,281
    Non-trainable params: 0
    _________________________________________________________________



```python
# compile the model
regressor.compile(optimizer=rms, loss='mean_squared_error',
                   metrics =['mean_absolute_error', 'mean_squared_error'])
```


```python
regressor.fit(X_train, Y_train, epochs=500, validation_split=0.2)
```
    Train on 73216 samples, validate on 18305 samples
    Epoch 1/500
    73216/73216 [==============================] - 5s 72us/sample - loss: 82631.6743 - mean_absolute_error: 21.8541 - mean_squared_error: 82631.0859 - val_loss: 37.4663 - val_mean_absolute_error: 1.6977 - val_mean_squared_error: 37.4663
    Epoch 2/500
    73216/73216 [==============================] - 5s 69us/sample - loss: 15537.8614 - mean_absolute_error: 11.3774 - mean_squared_error: 15537.8564 - val_loss: 6006.3956 - val_mean_absolute_error: 8.2804 - val_mean_squared_error: 6006.3979
    Epoch 3/500
    73216/73216 [==============================] - 5s 67us/sample - loss: 8787.8236 - mean_absolute_error: 7.5955 - mean_squared_error: 8787.8213 - val_loss: 17419.2455 - val_mean_absolute_error: 14.2364 - val_mean_squared_error: 17419.2461
    Epoch 4/500
    73216/73216 [==============================] - 5s 68us/sample - loss: 6664.4345 - mean_absolute_error: 6.2893 - mean_squared_error: 6664.4321 - val_loss: 11492.3081 - val_mean_absolute_error: 11.3387 - val_mean_squared_error: 11492.2998
    Epoch 5/500
    73216/73216 [==============================] - 5s 69us/sample - loss: 2706.5453 - mean_absolute_error: 4.7715 - mean_squared_error: 2706.5427 - val_loss: 6.2631 - val_mean_absolute_error: 1.1762 - val_mean_squared_error: 6.2631
    Epoch 50/500
    73216/73216 [==============================] - 5s 68us/sample - loss: 1.7080 - mean_absolute_error: 0.9173 - mean_squared_error: 1.7080 - val_loss: 1.8886 - val_mean_absolute_error: 0.9192 - val_mean_squared_error: 1.8886
    Epoch 100/500
    73216/73216 [==============================] - 5s 66us/sample - loss: 1.6543 - mean_absolute_error: 0.9233 - mean_squared_error: 1.6543 - val_loss: 1.5451 - val_mean_absolute_error: 0.9456 - val_mean_squared_error: 1.5451
    Epoch 200/500
    73216/73216 [==============================] - 5s 67us/sample - loss: 1.3497 - mean_absolute_error: 0.8743 - mean_squared_error: 1.3497 - val_loss: 1.3646 - val_mean_absolute_error: 0.8906 - val_mean_squared_error: 1.3646
    Epoch 300/500
    73216/73216 [==============================] - 5s 67us/sample - loss: 1.3755 - mean_absolute_error: 0.8867 - mean_squared_error: 1.3755 - val_loss: 1.3364 - val_mean_absolute_error: 0.8615 - val_mean_squared_error: 1.3364
    Epoch 400/500
    73216/73216 [==============================] - 5s 68us/sample - loss: 1.4052 - mean_absolute_error: 0.8894 - mean_squared_error: 1.4052 - val_loss: 1.4551 - val_mean_absolute_error: 0.8892 - val_mean_squared_error: 1.4551
    Epoch 500/500
    73216/73216 [==============================] - 5s 70us/sample - loss: 2.2173 - mean_absolute_error: 0.8802 - mean_squared_error: 2.2173 - val_loss: 1.3030 - val_mean_absolute_error: 0.8503 - val_mean_squared_error: 1.3030


    91521/91521 [==============================] - 3s 29us/sample - loss: 1.2819 - mean_absolute_error: 0.8452 - mean_squared_error: 1.2819
    Training Loss: 1.2818819236931775
    Training MAE: 0.84515953
    Training MSE: 1.2818828


    39224/39224 [==============================] - 1s 28us/sample - loss: 1.3134 - mean_absolute_error: 0.8548 - mean_squared_error: 1.3134
    Test Loss: 1.313426486112963
    Test MAE: 0.85482204
    Test MSE: 1.3134267

    R-squared value: 0.51397

### Run 4 with arch: [22, 32, 64, 64, 128, 64, 64 32, 1]
```python
regressor = Sequential()

regressor.add(Dense(32, activation='relu', input_dim=22))

regressor.add(Dense(64, activation='relu'))

regressor.add(Dense(64, activation='relu'))

regressor.add(Dense(128, activation='relu'))

regressor.add(Dense(64, activation='relu'))

regressor.add(Dense(64, activation='relu'))

regressor.add(Dense(32, activation='relu'))

regressor.add(Dense(1))
```
    /GPU:0 - NVIDIA Corporation GK210GL [Tesla K80]

```python
regressor.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 32)                736       
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                2112      
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_3 (Dense)              (None, 128)               8320      
    _________________________________________________________________
    dense_4 (Dense)              (None, 64)                8256      
    _________________________________________________________________
    dense_5 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_6 (Dense)              (None, 32)                2080      
    _________________________________________________________________
    dense_7 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 29,857
    Trainable params: 29,857
    Non-trainable params: 0
    _________________________________________________________________


```python
# compile the model
regressor.compile(optimizer=rms, loss='mean_squared_error',
                   metrics =['mean_absolute_error', 'mean_squared_error'])
```


```python
regressor.fit(X_train, Y_train, epochs=500, validation_split=0.2, batch_size=128)
```
    Train on 73216 samples, validate on 18305 samples
    Epoch 1/500
    73216/73216 [==============================] - 2s 29us/sample - loss: 36304.4227 - mean_absolute_error: 11.3972 - mean_squared_error: 36304.4414 - val_loss: 222.8230 - val_mean_absolute_error: 2.5017 - val_mean_squared_error: 222.8230
    Epoch 2/500
    73216/73216 [==============================] - 2s 23us/sample - loss: 590.6254 - mean_absolute_error: 2.8668 - mean_squared_error: 590.6252 - val_loss: 16.1485 - val_mean_absolute_error: 1.5224 - val_mean_squared_error: 16.1485
    Epoch 3/500
    73216/73216 [==============================] - 2s 23us/sample - loss: 136.5936 - mean_absolute_error: 1.9896 - mean_squared_error: 136.5937 - val_loss: 21.2147 - val_mean_absolute_error: 1.6377 - val_mean_squared_error: 21.2147
    Epoch 4/500
    73216/73216 [==============================] - 2s 22us/sample - loss: 103.9836 - mean_absolute_error: 1.5385 - mean_squared_error: 103.9836 - val_loss: 2.3227 - val_mean_absolute_error: 1.1636 - val_mean_squared_error: 2.3227
    Epoch 5/500
    73216/73216 [==============================] - 2s 22us/sample - loss: 78.1466 - mean_absolute_error: 1.5526 - mean_squared_error: 78.1466 - val_loss: 1.6615 - val_mean_absolute_error: 0.9808 - val_mean_squared_error: 1.6615
    Epoch 50/500
    73216/73216 [==============================] - 2s 22us/sample - loss: 3.4092 - mean_absolute_error: 0.9161 - mean_squared_error: 3.4092 - val_loss: 1.3815 - val_mean_absolute_error: 0.8751 - val_mean_squared_error: 1.3815
    Epoch 100/500
    73216/73216 [==============================] - 2s 22us/sample - loss: 1.3334 - mean_absolute_error: 0.8612 - mean_squared_error: 1.3334 - val_loss: 1.3525 - val_mean_absolute_error: 0.8787 - val_mean_squared_error: 1.3525
    Epoch 200/500
    73216/73216 [==============================] - 2s 22us/sample - loss: 1.3462 - mean_absolute_error: 0.8602 - mean_squared_error: 1.3462 - val_loss: 1.3354 - val_mean_absolute_error: 0.8684 - val_mean_squared_error: 1.3354
    Epoch 300/500
    73216/73216 [==============================] - 2s 22us/sample - loss: 1.2962 - mean_absolute_error: 0.8403 - mean_squared_error: 1.2962 - val_loss: 1.3546 - val_mean_absolute_error: 0.8503 - val_mean_squared_error: 1.3546
    Epoch 400/500
    73216/73216 [==============================] - 5s 68us/sample - loss: 1.4052 - mean_absolute_error: 0.8894 - mean_squared_error: 1.4052 - val_loss: 1.4551 - val_mean_absolute_error: 0.8892 - val_mean_squared_error: 1.4551
    Epoch 500/500
    73216/73216 [==============================] - 5s 70us/sample - loss: 2.2173 - mean_absolute_error: 0.8802 - mean_squared_error: 2.2173 - val_loss: 1.3030 - val_mean_absolute_error: 0.8503 - val_mean_squared_error: 1.3030

    91521/91521 [==============================] - 3s 30us/sample - loss: 1.2497 - mean_absolute_error: 0.8416 - mean_squared_error: 1.2497
    Training Loss: 1.2496539591773184
    Training MAE: 0.8415788
    Training MSE: 1.2496532

    39224/39224 [==============================] - 1s 30us/sample - loss: 1.2976 - mean_absolute_error: 0.8558 - mean_squared_error: 1.2976
    Test Loss: 1.2975662440006863
    Test MAE: 0.85583407
    Test MSE: 1.297566

    R-squared value: 0.52284
