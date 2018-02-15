# ML Encoders

Machine Learning encoders for feature transform and engineering.
These encoders implement the same API as ML models from `sklearn`, and expose the usual `fit`, `transform` and `fit_transform` methods.

## Setup

Simply install from `pip`:
```
pip install mlencoders
```

## Encoders

Below is the list of encoders available.

### Target Encoder

Also known as "Mean Encoder" or "Likelihood Encoder". See [this publication](https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf) for more background.

Allows to encode (possibly high cardinality) categorical features `X` into a continuous value `P(y | X)` where `y` is the target variable we wish to learn in our ML application.

See the following example to observe how the categorical variables are transformed:

```python
from sklearn.datasets import load_boston

import pandas as pd
import numpy as np

boston = load_boston()
y = pd.Series(boston.target)
X = pd.DataFrame(boston.data, columns=boston.feature_names)

enc = TargetEncoder(cols=['CHAS', 'RAD'])
X_encoded = enc.fit_transform(X, y)
```

### More to come!

## Requirements

* `pandas >= 0.22.0`
* `numpy >= 1.14.0`
