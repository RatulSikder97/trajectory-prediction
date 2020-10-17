# Trajectory path prediction

## Basic trajectory path prediction using RNN,LSTM

```code
Author: Ratul Sikder
Gmail: ratulsikder104@gmail.com

```

_Please load **trajectoryPred.ipynb** in Jupyter Notebook or Run **trajectoryPred.py**_

### Project target

Dataset contain X and Y coordinates of a projectile of Golf ball .Here target is to find future path of the projected object.

#### Datasets

Projectile initial velocity about _100ms<sup>-1</sup>_ <br>
two datasets:

- Initial 60 deg angle
- Initial 45 deg angle

### Approach:

Trajectory is the path of projectile motion of object. It can be categetoried as Time Series Analysis. So, in this peoject RNN-LSTM model is build with one Input layer, one LSTM hidden layer and one output layer. As activation function, here I use tanh function.

### Model

<ul>
    <li>Layer Quantity : 3</li>
    <li>Loss Function : Mean Squared Error</li>
    <li>Activation Fucntion: tanh(for all layer)</li>

</ul>

```console
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 10)                480
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 491
Trainable params: 491
Non-trainable params: 0
```

## Experiment

Here dataset with 45 deg initial angle is used.

- Dataset Shape (1443, 5)

#### Dataset split

- Training data size 966
- Testing data size 477

**Memory lookback for LSTM network 20**

_\* Finally, dataset with 60 deg initial angle is used for further validation_

## Result Analysis

### Actual path

![XY plot](/fig/xy-1.png)

### Output

![output_1](/fig/predict-1.png)

```console
Train Error Score: 1.31 RMSE
Test Error Score: 4.98 RMSE
```

<p style="color:#008891"><b>Too little error</b></p>

## I/O of another dataset

### Actual path

![XY plot](/fig/ac-2.png)

### Output

![output_1](/fig/pre-2.png)

```console
Train Score: 1.66 RMSE
Test Score: 6.55 RMSE
```

<p style="color:#008845"><b>Here also get about perfect result.</b></p>
