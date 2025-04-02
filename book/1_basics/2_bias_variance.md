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

# <i class="fa-solid fa-handshake"></i> Bias-Variance Tradeoff

Before we dive into the concept of bias, let’s briefly recap some theoretical concepts you learned about in the lecture. When we talk about fitting machine learning algorithms, we are referring to the process of estimating a function $f$ that best represents the relationship between the outcome and a set of labelled data (in supervised learning) or to uncover structural patterns in unlabelled data (in unsupervised learning). While the estimated function $\hat{f}$​ conveys important information about the data from which it was derived (the training data), our primary interest is in using this function to make accurate predictions for future cases in new, unseen data sets. 

The fundamental question in statistical learning is how well $\hat{f}$​ will perform on these future data sets, which brings us to the *bias-variance tradeoff*. Bias occurs when a model is too simple to capture the underlying complexities of the data, leading to systematic inaccuracies in its predictions. Variance measures how much the model's predictions fluctuate when trained on different subsets of the data.

```{admonition} Reminder
:class: note

- **Bias**: The error introduced when a model makes too simple assumptions about the data
- **Variance**: The sensitivity of the model to small changes in the training data
```

This closely relates to the example introduced in [](0_refresher). Let's have a another look and simulate some data with an underlying cubic polynomial. We can see that a linear regression does not capture the nuance of the cubic relationship in the data, while a 10th order model already overfits quite a lot:

```{code-block} ipython3
import numpy as np

x = np.linspace(-3, 3, 30)
y = (x**3 + np.random.normal(0, 15, size=x.shape)) / 10
```

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme()
np.random.seed(42)

x = np.linspace(-3, 3, 30)
y = (x**3 + np.random.normal(0, 15, size=x.shape)) / 10
df = pd.DataFrame({'x': x, 'y': y})

fig, ax = plt.subplots(1,3, figsize=(10,4))
sns.regplot(df, x="x", y="y", ax=ax[0], order=1)
sns.regplot(df, x="x", y="y", ax=ax[1], order=3)
sns.regplot(df, x="x", y="y", ax=ax[2], order=10)

titles = ["1st order model", "3rd order model", "10th order model"]
for a, title in zip(ax, titles):
    a.set(title=title)
    a.set_xlim(-3, 3)
    a.set_ylim(-4, 4)

plt.tight_layout()
```

If we look at the mean squared error (MSE), we see that it decreases with increasing model flexibility:

```{code-cell} ipython3
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

# Create data
np.random.seed(42)
x = np.linspace(-3, 3, 30).reshape(-1, 1)
y = (x**3 + np.random.normal(0, 15, size=x.shape)) / 10

# Run the models
mse = []
models = []
degrees = [1, 3, 10]

for degree in degrees:
  x_trans = PolynomialFeatures(degree=degree).fit_transform(x)
  model = sm.OLS(y, x_trans).fit()

  models.append(model)
  mse.append(np.mean(model.resid**2))

# Print results
print(f"Degree  Train MSE")
print(f"1       {mse[0]:.3f}")
print(f"3       {mse[1]:.3f}")
print(f"10      {mse[2]:.3f}")
```

However, if we evaluate the same models on new, unseen data, we see that the MSE is now increasing wit

```{code-cell} ipython3
:tags: [remove-input]

# Create data
np.random.seed(42)
x_train = np.linspace(-3, 3, 30)
y_train = (x_train**3 + np.random.normal(0, 15, size=x_train.shape)) / 10

np.random.seed(43)
x_test = np.linspace(-3, 3, 30)
y_test = (x_test**3 + np.random.normal(0, 15, size=x_test.shape)) / 10

# Create plot
fig, ax = plt.subplots(1,3, figsize=(10,4))
orders = [1, 3, 10]
titles = ["1st order model", "3rd order model", "10th order model"]
predictions = []

for a, order, title in zip(ax, orders, titles):
  coeffs = np.polyfit(x_train, y_train, order)
  x_fit = np.linspace(-5, 5, 400)
  y_fit = np.polyval(coeffs, x_fit)
  y_pred = np.polyval(coeffs, x_test)
  predictions.append(y_pred)

  sns.scatterplot(df, x="x", y="y", ax=a)
  a.plot(x_fit, y_fit)
  
  a.set(title=title)
  a.set_xlim(-3, 3)
  a.set_ylim(-4, 4)

plt.tight_layout()
plt.show()

# Calculate the test MSE
resids = [pred - y_test for pred in predictions]
mses = [np.mean(resid**2) for resid in resids]

print(f"Degree  Test MSE")
print(f"1       {mses[0]:.3f}")
print(f"3       {mses[1]:.3f}")
print(f"10      {mses[2]:.3f}")
```


One key issue is handling irreducible error and keeping training as well as test MSE in check. A model that is too closely fitted to the training data may resemble a tailor-made suit—perfectly customized for one person but unlikely to fit anyone else. Similarly, a model with a very low **training MSE** might appear to perform well, but its inability to generalize to new, unseen data often results in a high **test MSE**.


```{figure} figures/bias_variance.drawio.png
:alt: BiasVarianceTradeOff
:width: 300
:align: right

Bias-variance tradeoff
```

As you make your model more flexible, in other words taking more paremters into account, you are essentially **trading off between bias and variance**.

Initially, as the model becomes more flexible, its bias decreases faster than its variance increases. This occurs because a more flexible model can capture more complex patterns, thereby reducing bias. However, increased flexibility also makes the model more sensitive to fluctuations in the training data, which increases variance. Eventually, the reduction in bias stops compensating for the increase in variance, causing the test MSE to first decrease and then increase.

<br><br>


```{admonition} Summary
:class: tip

To achieve the best possible training MSE, we need to **minimize** both bias and variance simultaneously!
```

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz
display_quiz('quiz/bias_variance.json')
```
