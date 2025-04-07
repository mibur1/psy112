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

$$ y = b0 + b1*x1 + b2*x2 + ... + bp*xp + e $$


we do this:

$$ y = b0 + f1(x1) + f2(x2) + ... + fp(xp) + e $$


So instead of using fixed slope coefficients *bp**​ that assume a straight-line relationship, we replace them with flexible (possibly non-linear) **smooth functions** *fp*​ for each predictor!


```{admonition} GAMs
:class: note 

Importantly, GAMs learn non-linear relationships in the data such that: 
- The knots for spline functions are automatically selected
- The degree of flexibility of the smoothing functions (mostly splines) are automatically selected
- Splines from several predictors can be combined simultaneously.
```


```{code-cell} ipython3
:tags: ["remove-input"]
from jupyterquiz import display_quiz
display_quiz("quiz/GAM.json", shuffle_answers=False)
```
------------------------------------------------
## *Todays data - Diabetes dataset*
For today's practical demonstration, we will work with the well-known `Diabetes` dataset from `scikit-learn`. This dataset contains medical information collected from 442 diabetes patients, including:
- 10 baseline features measured at the beginning of the study:
    - age, sex, Body Mass Index (BMI), average blood pressure, six blood serum measurements (e.g. cholesterol, blood sugar, etc.)
- Target variable: A quantitative measure of disease progression one year after the baseline measurements were taken.

You can find more information here: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/descr/diabetes.rst.

```{code-cell} 
from sklearn import datasets

# load data
diabetes = datasets.load_diabetes(as_frame=True)

# define feature and target
X= diabetes.data
y= diabetes.target
```
As you're already familiar with from previous weeks, let's begin by splitting the data into training and test sets. This allows us to train the model on one portion of the data and evaluate its performance on unseen data — just like we did with other models before.

<iframe src="https://trinket.io/embed/python3/8753dba7854a" width="100%" height="356" frameborder="0" marginwidth="0" marginheight="0" allowfullscreen></iframe>

```{code-cell} ipython3
:tags: [remove-input]
from sklearn.model_selection import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=42)
```


To explore the relationships between each feature and the target variable, we plot each predictor against the disease progression outcome. These scatter plots with simple linear regression lines help us visually assess whether the relationship between a feature and the target is linear or if we need a more flexible model approach.

```{code-cell} ipython3
:tags: ["remove-input"]

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Load data
diabetes = datasets.load_diabetes(as_frame=True)
X = diabetes.data
y = diabetes.target

# Combine into a single DataFrame for easy plotting
df = X.copy()
df['target'] = y

features_to_plot = X.columns

# Set up the plot grid
num_features = len(features_to_plot)
cols = 3
rows = (num_features + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
axes = axes.flatten()

# Plot each feature (except 'sex') against the target
for i, col in enumerate(features_to_plot):
    sns.regplot(x=col, y='target', data=df, ax=axes[i], scatter_kws={'s': 10}, line_kws={'color': 'red'})
    axes[i].set_title(f"{col} vs target")

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

```
From the plots, we can see that a simple linear regression line often does not fully capture the complexity of the relationships between the features and the target variable. This suggests that a more flexible modeling approach—like GAMs—may be better suited for this dataset.- not for all feautes linear regression seems to 
Lets try it.

------------------------------------------------
## GAM 
`pygam` is a user-friendly Python library for fitting Generalized Additive Models (GAMs), allowing flexible, smooth modeling of nonlinear relationships without requiring manual setup of spline bases or complex statistical details.

In `pygam`we have 3 different building blocks for our GAM formula:
- `s()` - stands for **spline**, and is used when you suspect a nonlinear relationship between a feature and the target. pygam will learn a smooth curve from the data.
- `l()` -  the **linear** term, used when the relationship appears to be approximately straight. It keeps the model simple and more interpretable.
- `f()` -   stands for **factor**, and is used when the feature is categorical (e.g. binary variables like sex, or multi-class variables).


Look at the plot above:
  - Are there any features that could be modeled using linear regression?
  - Are there any categorical features?

```{code-cell}
from sklearn.datasets import load_diabetes
from pygam import LinearGAM, s, f, l 

# Create the GAM model
gam = LinearGAM(
    s(X.columns.get_loc('age')) +                   # age: smooth term
    f(X.columns.get_loc('sex')) +                   # sex: factor
    s(X.columns.get_loc('bmi')) +                   # bmi: smooth
    s(X.columns.get_loc('bp')) +                    # bp: smooth
    s(X.columns.get_loc('s1')) +                    # s1: smooth
    s(X.columns.get_loc('s2')) +                    # s2: smooth
    s(X.columns.get_loc('s3')) +                    # s3: smooth
    s(X.columns.get_loc('s4')) +                    # s4: smooth
    l(X.columns.get_loc('s5')) +                    # s5: linear
    s(X.columns.get_loc('s6'))                      # s6: smooth 
) 

# Fit the model
gam.fit(X_train, y_train)

# Print summary 
gam.summary()
```
The GAM model explains about 64% of the variance, showing a good fit. Key predictors include bmi, sex, s5, which show significant effects. Some features like age, s1, s2, s3, s4 and s6 appear less important. Note that p-values may be slightly optimistic due to smoothing estimation.

To evaluate how well our model generalizes to unseen data, we split the data into training and test sets. This allows us to train the model on one part of the data and evaluate its performance on a separate, unseen portion.
As a comparison, we apply the same train/test evaluation to a linear regression model to see how it performs on the same test data.

```{code-cell}
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# LINEAR REGRESSION
# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)
score_lr = r2_score(y_test, y_pred_lr)

# GAM
# predict on test set
y_pred= gam.predict(X_test)

# Evaluate performance
score_gam = r2_score(y_test, y_pred)

# printing results
print("GAM CV scores:", score_gam)
print("Linear CV scores:", score_lr)
```
As we can see, our GAM model **does not perform as well on unseen** data. While the training data showed a strong R² of 0.6471, the test performance dropped to 0.38, which is a typical sign of **overfitting**. In contrast, the simpler linear regression model achieved a higher R² of 0.48 on the same test data.
However, one of the strengths of GAMs is their **flexibility**. We can easily adjust the model—for example, by changing how we treat specific features (e.g. linear instead of spline, or vice versa)—to improve generalization. 


**Now it's your turn:** Head to the exercise section and try to improve the GAM's performance on unseen data. Use the initial feature-target plots as a guide to decide which features might benefit from a linear, smooth, or categorical treatment.