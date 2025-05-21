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

# <i class="fa-solid fa-lightbulb"></i> Naïve Bayes

https://animlbook.com/classification/naive_bayes/index.html

https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html

Many of the previously introduced machine learning models for regression and classification tasks are based on linear models. 


Naïve Bayes applies Bayes' theorem under the assumption that predictors are independent within each class:

$$
P(Y = k \mid X = x) = \frac{f_k(x) \cdot \pi_k}{\sum_{l=1}^{K} \pi_l \cdot f_l(x)}
$$

## Independence Assumption

In Naïve Bayes, the joint density function of the predictors is decomposed as:

$$
f_k(x) = f_{k1}(x_1) \cdot f_{k2}(x_2) \cdot \ldots \cdot f_{kp}(x_p)
$$

This allows the method to work even when the predictors follow different distributions (Gaussian, Binomial, etc.).

## Naïve Bayes in Python

We will use the same simulated data as in the LDA/QDA session.

1. Generate the data

```{code-cell} ipython3
import numpy as np
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Generate synthetic data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
                           n_redundant=0, n_classes=2, n_clusters_per_class=1, 
                           random_state=42);
```

2. Fit the model

```{code-cell} ipython3
nb = GaussianNB()
nb.fit(X, y);
```

3. Plot the predicted distributions and classification report

```{code-cell} ipython3
import seaborn as sns
import matplotlib.pyplot as plt

# Plot distributions
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots()

ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='bwr')
ax.set_title('Naïve Bayes', size=14)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xg = np.linspace(x_min, x_max, 60)
yg = np.linspace(y_min, y_max, 40)
xx, yy = np.meshgrid(xg, yg)
Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

for label, color in enumerate(['blue', 'red']):
    mask = (y == label)
    mu, std = X[mask].mean(0), X[mask].std(0)
    P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
    Pm = np.ma.masked_array(P, P < 0.03)
    ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.4,
                  cmap=color.title() + 's')
    ax.contour(xx, yy, P.reshape(xx.shape),
               levels=[0.01, 0.1, 0.5, 0.9],
               colors=color, alpha=0.2)
plt.show()

# Print classification report
print(classification_report(y, nb.predict(X)))
```