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

# <i class="fa-solid fa-gear"></i> Support Vector Machines

WORK IN PROGRESS

After a brief excursion into generative models such as [discriminant analysis](5_LDA_QDA) or [Naïve Bayes](6_Naive_Bayes), we will now again discuss a discriminative family of models: Support Vector Machines (SVM). SVMs are powerful supervised learning models used for classification and regression tasks. When used for classification, they are called Support Vector Classifiers (SVC).

Let's consider some simulated classification data:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import make_classification
sns.set_theme(style="darkgrid")

X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, 
                           n_clusters_per_class=1, class_sep=2.0, random_state=0)

fig, ax = plt.subplots()
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=ax)
ax.set(xlabel="Feature 1", ylabel="Feature 2")
ax.legend(labels=["Class 0", "Class 1"]);
```

How



- **Hyperplane**: A decision boundary that separates classes. In p dimensions, it is a p−1 dimensional flat affine subspace, given by the equation:
  $\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p = 0$
  The vector $\beta = (\beta_1, \beta_2, \dots, \beta_p)$ is the normal vector.
- **Separating Hyperplane**: A hyperplane that correctly separates the data by class label.
- **Margin**: The (perpendicular) distance between the hyperplane and the closest training points. A maximal margin classifier chooses the hyperplane that maximises this margin.
- **Support Vectors**: Observations closest to the decision boundary. They define the margin and the classifier.
- **Soft Margin**: A method used when the data is not linearly separable. Allows some observations to violate the margin. Controlled via the hyperparameter $C$.
- **Kernel Trick**: Implicitly maps data into a higher-dimensional space to make it linearly separable using functions like polynomial or RBF (Gaussian) kernels.

## When to Use SVC

- When the number of features is large relative to the number of samples
- When classes are not linearly separable
- When a robust and generalisable classifier is needed

## Linear vs Nonlinear SVC

- **Linear SVC**: Suitable when data is linearly separable or when using a linear decision boundary is sufficient.
- **Nonlinear SVC**: Use kernel methods when data exhibits nonlinear patterns.

## Example 1: Linear SVC on Linearly Separable Data

```{code-cell} ipython3
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, class_sep=2.0)

clf = SVC(kernel='linear')
clf.fit(X, y)

w = clf.coef_[0]
a = -w[0] / w[1]
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max())
y_vals = a * x_vals - (clf.intercept_[0]) / w[1]

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.plot(x_vals, y_vals, 'k-')
plt.title('Linear SVC Decision Boundary')
plt.show()
```

## Example 2: Nonlinear Classification with RBF Kernel

```{code-cell} ipython3
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
```

## Multiclass Classification

SVMs are inherently binary classifiers but can be extended:

* **One-vs-One**: $\binom{K}{2}$ classifiers for each pair of classes.
* **One-vs-All**: K classifiers, each comparing one class against the rest.

## Choosing Hyperparameters

* `C`: Regularisation parameter; trade-off between margin width and classification error.
* `kernel`: `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`, or custom.
* `gamma`: Defines the influence of a training example; affects RBF, polynomial, and sigmoid kernels.

Hyperparameters should be tuned using cross-validation to balance bias and variance.

## Summary

Support Vector Classifiers are a robust and versatile tool for classification tasks. The key ideas are rooted in geometry—finding the optimal hyperplane that separates data with maximum margin. With the use of kernels, SVMs extend effectively to non-linear decision boundaries and multiclass problems.