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

Machine learning systems are often just automated ways of reflecting patterns in the data they were trained on. The main goal is to keep the Mean Squared Error (MSE) as low as possible. While this might sound simple, it is actually quite challenging to find the perfect balance between generalization and accuracy. This challenge is defined by the **bias-variance trade-off**.

```{admonition} Remember
:class: note

**Bias** is defined as the error that is introduced by approximating a real-life problem, which might be extremly complex, by a much simplet model. 
 
 **Variance** refers to the amount by which the approximated function would change if we estimated it using a different training data set. 
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
