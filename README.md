# Quantile Regression Neural Network

This package is based on the paper, [An **improved quantile regression neural network** for probabilistic load forecasting](https://ieeexplore.ieee.org/abstract/document/8419220/), [W Zhang](https://scholar.google.com/citations?user=aanT6TIAAAAJ&hl=en&oi=sra).

## Usage

```python
from qrnn import get_model, Qloss, qloss
from keras.callbacks import *
import numpy as np

# Generate the synthetic data
x1 = np.sin(np.arange(0, 9, 0.01))
x2 = np.cos(np.arange(0, 9, 0.01))
x3 = x1**2
x4 = (x1+x2)/2

Xtrain = np.vstack((x2, x3, x4)).T #(900, 3)
Ytrain = np.array([x1]*99).T #(900, 1)

# Parameters
input_dim = 3
num_hidden_layers = 2
num_units = [200, 200]
act = ['relu', 'relu']
dropout = [0.1, 0.1]
gauss_std = [0.3, 0.3]

# Get model
model = get_model(input_dim, num_units, act, dropout, gauss_std, num_hidden_layers)
print(model.summary())

# Train
early_stopping = EarlyStopping(monitor='val_qloss_score', patience=5)
model.compile(loss=qloss, optimizer='adam', metrics=[Qloss()])
model.fit(x=Xtrain, y=Ytrain, 
          epochs=10, 
          validation_split=0.2, 
          batch_size=64, 
          shuffle=True, 
          callbacks=[early_stopping]
         )
```
