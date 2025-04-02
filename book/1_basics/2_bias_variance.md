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

Before we dive into the concept of bias, let’s briefly recap some theoretical concepts you learned about in the lecture. When we talk about fitting machine learning algorithms, we are referring to the process of estimating a function $f$ that best represents the relationship between the outcome and a set of labelled data (in supervised learning) or to uncover structural patterns in unlabelled data (in unsupervised learning). While the estimated function $\hat{f}$​ conveys important information about the data from which it was derived (the training data), our primary interest is in using this function to make accurate predictions for future cases in new, unseen data sets. A fundamental question in statistical learning is how well $\hat{f}$​ will perform on these future data sets, which brings us to the *bias-variance tradeoff* — a key factor in understanding how well our model generalizes.

```{admonition} Reminder
:class: note

**Bias** refers to the error introduced when a model makes simplifying assumptions about the real-world problem. It occurs when the model is too simple to capture the underlying complexities of the data, leading to systematic inaccuracies in predictions.

**Variance** refers to the sensitivity of the model to small changes in the training data. It measures how much the model's predictions fluctuate when trained on different subsets of the data.
```

```{margin}
Overfitting: occurs when a function is too closely aligned to the training data set, also catching the noise 
```

One key issue is handling irreducible error and keeping training as well as test MSE in check. A model that is too closely fitted to the training data may resemble a tailor-made suit—perfectly customized for one person but unlikely to fit anyone else. Similarly, a model with a very low **training MSE** might appear to perform well, but its inability to generalize to new, unseen data often results in a high **test MSE**.


```{image} figures/bias_variance.drawio.png
:alt: BiasVarianceTradeOff
:width: 33%
:align: right
```

As you make your model more flexible, in other words taking more paremters into account, you are essentially **trading off between bias and variance**.

Initially, as the model becomes more flexible, its bias decreases faster than its variance increases. This occurs because a more flexible model can capture more complex patterns, thereby reducing bias. However, increased flexibility also makes the model more sensitive to fluctuations in the training data, which increases variance. Eventually, the reduction in bias stops compensating for the increase in variance, causing the test MSE to first decrease and then increase.


```{admonition} Summary
:class: tip

To achieve the best possible training MSE, we need to **minimize** both bias and variance simultaneously!
```

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz
display_quiz('quiz/bias_variance.json')
```
