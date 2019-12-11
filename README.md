# Quantile Regression Neural Network

This package is based on the paper, [An **improved quantile regression neural network** for probabilistic load forecasting](https://ieeexplore.ieee.org/abstract/document/8419220/), [W Zhang](https://scholar.google.com/citations?user=aanT6TIAAAAJ&hl=en&oi=sra).

## Usage

```python
from qrnn import get_model
import numpy as np

# Generate the synthetic data
x1 = np.sin(np.arange(0, 9, 0.01))
x2 = np.cos(np.arange(0, 9, 0.01))
x3 = x1**2
x4 = (x1+x2)/2

Xtrain = np.vstack((x2, x3, x4)).T
Ytrain = np.array([x1]*99).T

# Parameters
input_dim = 3
num_hidden_layers = 1
num_units = [200]
act = ['relu']
gauss_std = 0.3

# Get model
model = get_model(input_dim, num_units, act, gauss_std, num_hidden_layers)

# Train
model.fit(x=Xtrain, y=Ytrain, epochs=10)
```
