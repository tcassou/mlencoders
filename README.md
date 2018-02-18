# ML Encoders

[![Build Status](https://travis-ci.org/tcassou/mlencoders.svg?branch=master)](https://travis-ci.org/tcassou/mlencoders)

Machine Learning encoders for feature transformation & engineering: target encoder, weight of evidence.
These encoders implement the same API as ML models from `sklearn`, and expose the usual `fit`, `transform` and `fit_transform` methods.

Available encoders:
* Target Encoder (a.k.a. likelihood encoder, or mean encoder)
* Weight of Evidence

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
from mlencoders.target_encoder import TargetEncoder
import pandas as pd

boston = load_boston()
y = pd.Series(boston.target)
X = pd.DataFrame(boston.data, columns=boston.feature_names)

enc = TargetEncoder(cols=['CHAS', 'RAD'])
X_encoded = enc.fit_transform(X, y)
```

### Weight of Evidence
See [this nice article](https://multithreaded.stitchfix.com/blog/2015/08/13/weight-of-evidence/) to learn about **Information Value (IV)** and **Weight of Evidence (WOE)**.

For a task with a binary target `Y` (e.g. binary classification), allows to encode categorical features `X` into a continuous value `WOE = log[ P(X=X_i | Y=1) / P(X=X_i | Y=0) ]`.

```python
... # load the same dataset as above

from mlencoders.weight_of_evidence_encoder import WeightOfEvidenceEncoder

enc = WeightOfEvidenceEncoder(cols=['CHAS', 'RAD'])
X_encoded = enc.fit_transform(X, y)
```

### More to come!

## Saving encoder state
In case you are planning to fit your encoders offline, and use them online at prediction time, you can easily save their state in a file and load it later on.

```python
# Offilne: fitting the encoder to data (X, y) and storing state
...
enc = TargetEncoder(some, parameters)
enc.fit(X, y)
enc.save_as_object_file('your_file_name')

# Online: loading your encoder and encoding new data X_new
...
enc = TargetEncoder()   # no parameters are needed here, they will be loaded automatically
enc.load_from_object_file('your_file_name')
enc.transform(X_new)
```

## Requirements

* `pandas >= 0.22.0`
* `numpy >= 1.14.0`
