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

# <i class="fa-solid fa-circle-plus"></i> Generalized Additive Models

Generalized Additive Models (GAMs) offer a powerful and flexible extension to traditional linear models by allowing for **non-linear, additive relationships** between each predictor and the outcome. Unlike standard linear regression, which assumes a strictly linear association between predictors and the response variable, GAMs replace each linear term with a smooth function, enabling the model to better **capture complex patterns** in the data. Thanks to their additive structure, each predictor contributes independently to the model, making it much easier to interpret the effect of each variable. 

So instead of using the standard linear function
$$ y = b0 + b1*x1 + b2*x2 + ... + bp*xp + e
$$

we do this:
$$ y = b0 + f1(x1) + f2(x2) + ... + fp(xp) + e $$


So instead of using fixed slope coefficients *bp**​ that assume a straight-line relationship, we replace them with flexible (possibly non-linear) **smooth functions** *fp*​ for each predictor!

Some common ways to estimate these smooth functions are:
- Splines
- Polynomial functions
- Local regression


```{admonition} GAMs
:class: note 

Importantly, GAMs learn non-linear relationships in the data such that: 
- The knots for spline functions are automatically selected
- The degree of flexibility of the smoothing functions (mostly splines) are automatically selected
- Splines from several predictors can be combined simultaneously.
```
