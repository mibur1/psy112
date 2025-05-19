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

# <i class="fa-solid fa-divide"></i> LDA & QDA

https://islp.readthedocs.io/en/latest/labs/Ch04-classification-lab.html


## Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis (LDA) applies a two-step approach to model the joint probability of \(X\) and \(Y\):

1. **Step 1:** Model the distribution of the predictors \(X\) separately for each response class indicated by \(Y\).
2. **Step 2:** Use Bayes' theorem to calculate estimates for the posterior probability:

$$
P(Y = k \mid X = x) = \frac{P(X = x \mid Y = k) \cdot P(Y = k)}{P(X = x)}
$$


### Assumptions
- Observations within each class are drawn from a multivariate normal (Gaussian) distribution.
- Each class has a specific mean vector.
- The covariance matrix is common to all \(K\) classes.


### LDA for \(p = 1\)
The discriminant function for one predictor (\(p = 1\)) is given by:

$$
\delta_k(x) = x \cdot \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + \log(\pi_k)
$$

Here:
- \(\pi_k\) is the prior probability of class \(k\).
- \(\mu_k\) is the mean of observations in class \(k\).
- \(\sigma^2\) is the pooled variance across classes.


### LDA for \(p > 1\)
When \(X\) is a vector, the discriminant function becomes:

$$
\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log(\pi_k)
$$


## Quadratic Discriminant Analysis (QDA)

QDA is similar to LDA, but it relaxes the assumption of a common covariance matrix for all classes. Instead:
- Each class has its own covariance matrix.
- QDA is more flexible but requires estimating more parameters than LDA.


## LDA and QDA in Python

We here show how LDA and QDA can be implemented in Python using `sklearn`. FOr this, we use artifical data for classification:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
                           n_redundant=0, n_classes=2, n_clusters_per_class=1, 
                           random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.title('Generated Data')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2");
```

Fitting the model is straightforward. However, please have a look at the [documentation](https://scikit-learn.org/stable/api/sklearn.discriminant_analysis.html) for additional options such as the specific solver.

```{code-cell} ipython3
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, \
                                          QuadraticDiscriminantAnalysis as QDA

lda = LDA()
lda.fit(X, y)

qda = QDA()
qda.fit(X, y);
```

We can then plot the decision boundary and print the classification report:

```{code-cell} ipython3
from sklearn.metrics import classification_report

def plot_decision_boundary(model, X, y, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

# Plot decision boundaries
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].set_title('LDA Decision Boundary')
plot_decision_boundary(lda, X, y, ax[0])

ax[1].set_title('QDA Decision Boundary')
plot_decision_boundary(qda, X, y, ax[1])
plt.show()

# Print classification report
print('LDA Classification Report:')
print(classification_report(y, lda.predict(X)))

print('QDA Classification Report:')
print(classification_report(y, qda.predict(X)))
```