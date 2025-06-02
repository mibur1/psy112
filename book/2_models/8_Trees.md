---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
myst:
  substitutions:
    ref_test: 1
---

# <i class="fa-solid fa-tree"></i> Decision Trees

https://mlu-explain.github.io/decision-tree/

https://animlbook.com/classification/trees/index.html

https://mlu-explain.github.io/random-forest/



```{code-cell}ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_validate, KFold)
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
```

We will work with two datasets:

- Breast Cancer (binary classification)
  - Features: 30 real-valued measurements of tumors
  - Target: malignant (1) vs. benign (0)
- Diabetes (continuous regression)
  - Features: 10 baseline variables (age, sex, BMI, …)
  - Target: a quantitative measure of disease progression


```{code-cell}ipython3
X = [[0, 0], [1, 1]]
Y = [0, 1]

clf = DecisionTreeClassifier()
clf = clf.fit(X, Y)
```