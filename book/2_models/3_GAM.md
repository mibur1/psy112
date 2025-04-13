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
#from jupyterquiz import display_quiz
#display_quiz("quiz/GAM.json", shuffle_answers=False)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


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
From the plots, we can see that a simple linear regression line often does not fully capture the complexity of the relationships between the features and the target variable. This suggests that a more flexible modeling approach—like GAMs—may be better suited for this some of this features.


Lets try it.

------------------------------------------------
## GAM 
In theory, building a Generalized Additive Model (GAM) in Python should be straightforward. There are even libraries specifically created for this purpose, such as:
- [pygam](https://pygam.readthedocs.io/en/latest/) is user-friendly, but it does not work with newer versions of `numpy` and `scikit-learn` (due to outdated dependencies and lack of active maintenance)
- [generalized-additive-models]("https://github.com/tommyod/generalized-additive-models/tree/main") promising newer project, but it is still experimental, has an unstable API, and depends on specific versions of libraries


As a result, both tools can cause unexpected errors, environment conflicts, and difficult debugging. To avoid dependency issues and gain more control, we’ll take a hybrid approach:

- we use `patsy` to define spline basis functions 
- we use `sklearn`s `Linear Regression`to fit the model

```{margin}
B-splines are flexible, piecewise-polynomial curves
``` 
This gives us full control over the model terms, and allows us to focus on each *f(x)* function individually using B-splines with independently adjustable degrees of freedom (`df`). 

### What does `df` means in Splines ?
The `df` parameter controls the flexibility of the spline. It roughly determines how many bends or "wiggles" the function can make:
- Low `df` --> smoother, more rigid curve
- High `df` --> more flexible


You can think of it as: *How many independent patterns is the model allowed to detect?*. As always, be mindful of the risk of overfitting when the flexibility is too high.

```{code-cell} python3
:tags: ["remove-input"]

import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrix

# Wertebereich für BMI
X = np.linspace(df["bmi"].min(), df["bmi"].max(), 200)

# df-Werte, die wir vergleichen wollen
dfs = [3, 6, 15]

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

for i, df_val in enumerate(dfs):
    # Designmatrix berechnen
    splines = dmatrix(f"bs(x, df={df_val}, degree=3, include_intercept=False)", 
                      {"x": X}, return_type='dataframe')
    
    # Zeichnen
    for col in splines.columns:
        axes[i].plot(X, splines[col])
    axes[i].set_title(f"B-Splines (df={df_val})")
    axes[i].set_xlabel("BMI")
    axes[i].grid(True)

axes[0].set_ylabel("Basis Function Value")
plt.suptitle("Comparison of B-spline Basis Functions (on BMI)", fontsize=14)
plt.tight_layout()
plt.show()
``` 
This plot shows B-spline basis functions for the feature `BMI`. Each curve is active in a local region and **contributes to the prediction only in that area**. The functions overlap to ensure smooth transitions. The model learns how much to weight each of them.
Together, they form the final smooth prediction curve. As you can see, the number of curves depends on the `df`.


Below you can see how the spline-based function looks when applied to our data. A lower `df` results in a smoother, simpler fit, while a higher `df` allows the model to capture more complex patterns

```{code-cell} ipython3
:tags: ["remove-input"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from patsy import dmatrix

# Load data
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame.copy()

# Sort data for smoother plotting
df_sorted = df.sort_values("bmi").reset_index(drop=True)
X_vals = df_sorted["bmi"].values.reshape(-1, 1)
y_vals = df_sorted["target"].values

# Plot settings
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
df_values = [3, 6, 15]

for ax, df_val in zip(axes, df_values):
    # Build spline design matrix
    X_spline = dmatrix(
        f"bs(bmi, df={df_val}, degree=3, include_intercept=False)",
        {"bmi": X_vals.ravel()},
        return_type="dataframe"
    )

    # Fit linear model on spline-expanded features
    model = LinearRegression()
    model.fit(X_spline, y_vals)
    y_pred = model.predict(X_spline)

    # Plot
    sns.scatterplot(x=X_vals.ravel(), y=y_vals, alpha=0.3, ax=ax)
    ax.plot(X_vals.ravel(), y_pred, color="red", label=f"Spline Fit (df={df_val})")
    ax.set_title(f"Spline fit on BMI (df={df_val})")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Target" if df_val == 3 else "")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

```

### GAM coding
We we have 3 different building blocks for our GAM formula:
- `bs(variable, df=..)` - this is our B-spline basis expansion. It transforms a single variable into **multiple features** that represent smooth, piecewise polynomial functions. It allows the model to learn a smooth curve from the data, enabling flexibility in capturing **complex non-linear patterns** 
- `C(variable)` - is used when the feature is a categorical variable (e.g., binary features like sex or multi-class features). 
- `variable` - this is the **linear term**, which includes the variable in its raw, numeric form. The model will fit a straight, linear line for it.


Look at the plot from the beginning:
  - Are there any features that could be modeled using linear regression?
  - Are there any categorical features?

We start by defining our formula. 
```{code-cell}
# formula with building blocks independetly for each feature
formula = (
    "bs(age, df=6) + "           # smooth
    "C(sex) + "                  # categorical
    "bs(bmi, df=6) + "           # smooth
    "bs(bp, df=3) + "            # smooth
    "bs(s1, df=6) + "
    "bs(s2, df=6) + "
    "bs(s3, df=6) + "
    "bs(s4, df=6) + "
    "s5 + "                      # linear
    "bs(s6, df=6)"               # smooth
)
```
Next, we create a design matrix using `patsy`, which transforms our raw data into the format needed for a GAM.

```{code-cell}
from patsy import dmatrix

# combine into one trainings set for design matrix creation
train_data= X_train.copy()
train_data["target"]= y_train

# Build design matrix from training data
X_train_design = dmatrix(formula, data=train_data, return_type="dataframe")
```
Now we fit a simple linear regression model — but it's not that simple anymore:
Thanks to the spline terms, it behaves like a GAM and can capture nonlinear relationships.

```{code-cell}
from sklearn.linear_model import LinearRegression

# Fit model
model = LinearRegression()
model.fit(X_train_design, y_train)
```
Finally, we evaluate the model's performance on the training data using the R² score,
which tells us how much of the variance in the target the model explains. 

```{code-cell}
from sklearn.metrics import r2_score

# R² score 
r_squared = model.score(X_train_design, y_train)
print(f"R² - trainings data: {r_squared:.3f}")
``` 

The GAM model explains about 59% of the variance, showing a good fit. 

To evaluate how well our model generalizes to unseen data, we split the data into training and test sets. This allows us to train the model on one part of the data and evaluate its performance on a separate, unseen portion. 
As a comparison, we apply the same train/test evaluation to a linear regression model to see how it performs on the same test data.

```{code-cell}
from sklearn.metrics import r2_score

# --- GAM ---
test_data = X_test.copy()
test_data["target"] = y_test
# defining the matrix
X_test_design = dmatrix(formula, data=test_data, return_type="dataframe")

# GAM Predictions
y_pred_gam = model.predict(X_test_design)
score_gam = r2_score(y_test, y_pred_gam)

# --- LINEAR REGRESSION ---
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Linear Regression Prediction
y_pred_lr = linear_model.predict(X_test)
score_lr = r2_score(y_test, y_pred_lr)

# --- PRINT COMPARISON ---
print(f"GAM R²on test data:     {score_gam:.3f}")
print(f"Linear R² on test data:  {score_lr:.3f}")

```
As we can see, our GAM model **does not perform as well on unseen** data. While the training data showed a strong R² of 0.59, the test performance dropped to 0.44, which is a typical sign of **overfitting**. In contrast, the simpler linear regression model achieved a higher R² of 0.48 on the same test data.
However, one of the strengths of GAMs is their **flexibility**. We can easily adjust the model—for example, by changing how we treat specific features (e.g. linear instead of spline, or vice versa)—to improve generalization. 


**Now it's your turn:** Head to the exercise section and try to improve the GAM's performance on unseen data. Use the initial feature-target plots as a guide to decide which features might benefit from a linear, smooth, or categorical treatment.
