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

# <i class="fa-solid fa-repeat"></i> Recap: Regression Models

$R^2$
$R^2$
$R^2$

One of the the most important concepts you learned about in the [psy111 seminar](https://mibur1.github.io/psy111) were (linear) regression models. Let's quickly recap this concept and how to implement it in Python.

Have a look at the following code, which creates some simulated data. Can you guess from the code, what the underlying pattern is?

```{code-block} ipython3
import numpy as np

x = np.linspace(-5, 5, 30)
y = (x**3 + np.random.normal(0, 15, size=x.shape)) / 50
```

<details>
<summary><strong>Click to reveal the plot</strong></summary>

Here you can see the data in a scatterplot, with a linear regression model fitted to it. Do you think it fits the data well?

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Generate sample data
np.random.seed(42)
x = np.linspace(-5, 5, 60)
indices = np.sort(np.random.choice(np.arange(60), size=30, replace=False))
x = x[indices]
y = (x**3 + np.random.normal(0, 15, size=x.shape)) / 50
df = pd.DataFrame({'x': x, 'y': y})

# Create scatter trace for the raw data
scatter = go.Scatter(
    x=df['x'],
    y=df['y'],
    mode='markers',
    marker=dict(
        size=10,
        color='lightgrey',
        line=dict(color='gray', width=2)
    ),
    name='Data'
)

# Fit a linear regression model (polynomial order 1)
coeffs = np.polyfit(df['x'], df['y'], 1)
x_fit = np.linspace(-5, 5, 400)
y_fit = np.polyval(coeffs, x_fit)

# Create regression trace with a thicker blue line
regression = go.Scatter(
    x=x_fit,
    y=y_fit,
    mode='lines',
    name='Model',
    line=dict(width=3, color='#4c72b0')
)

# Combine the traces into one figure
data = [scatter, regression]

# Define layout without annotations or sliders
layout = go.Layout(
    xaxis=dict(title="x", range=[-5.5, 5.5], tickfont=dict(size=14), fixedrange=True, gridwidth=1, zerolinewidth=1),
    yaxis=dict(title="y", range=[-3, 3], tickfont=dict(size=14), fixedrange=True, gridwidth=1, zerolinewidth=1),
    legend=dict(font=dict(size=14)),
    margin=dict(l=10, r=10, t=30, b=20),
)
fig = go.Figure(data=data, layout=layout)
fig
```
</details>
<br>

Let's have a closer look at the model statistics:

```{code-cell} ipython3
import statsmodels.formula.api as smf

model = smf.ols("y ~ x", data=df).fit()
print(model.summary())
```

You can see that the linear regression has an R² of 0.753, which means that our model explains 75% of the variance in our data. That's pretty good! But I'm sure we can do better. After all, life is more complicated than just a straigth line, no? <sub>(and we also know that the underlying data is a 3rd order polynomial)</sub>

```{code-cell} ipython3
:tags: [remove-input]

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Suppress polyfit conditioning warnings
warnings.filterwarnings("ignore", message=".*Polyfit may be poorly conditioned.*")

# Generate sample data
np.random.seed(42)
x = np.linspace(-5, 5, 60)
indices = np.sort(np.random.choice(np.arange(60), size=30, replace=False))
x = x[indices]
y = (x**3 + np.random.normal(0, 15, size=x.shape)) / 50
df = pd.DataFrame({'x': x, 'y': y})

# Scatter trace for the raw data with updated marker properties
scatter = go.Scatter(x=df['x'], y=df['y'], mode='markers',
                     marker=dict(size=10, color='lightgrey', line=dict(color='gray', width=2)), name='Data') 

# Generate regression curves for polynomial orders 1 through 30 with updated line properties
regression_traces = []
r2_list = []
x_fit = np.linspace(-5, 5, 400)
for order in range(1, 31):
    coeffs = np.polyfit(df['x'], df['y'], order)
    y_fit = np.polyval(coeffs, x_fit)
    trace = go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Model', visible=False,
                       line=dict(width=3, color='#4c72b0'))
    regression_traces.append(trace)

    y_pred = np.polyval(coeffs, df['x'])
    r2 = 1 - np.sum((df['y'] - y_pred) ** 2) / np.sum((df['y'] - df['y'].mean()) ** 2)
    r2_list.append(r2)

# Combine data: always show the scatter trace, and one regression trace at a time (set first one visible)
data = [scatter] + regression_traces
data[1]['visible'] = True

# Create slider steps
steps = []
for i in range(30):
    vis = [True] + [False] * 30
    vis[i + 1] = True
    step = dict(
        method="update",
        args=[{"visible": vis},
              {"annotations": [dict(x=0.02, y=0.98, xref="paper", yref="paper", text=f"Model R² = {r2_list[i]:.3f}", 
                                    showarrow=False, font=dict(size=18, color="gray"), align="left")]}],
        label=str(i + 1)
    )
    steps.append(step)

# Define the slider
sliders = [dict(active=0, currentvalue={"prefix": "Order of the polynomial regression model: "}, pad={"t": 30}, steps=steps)]

# Define the layout with
layout = go.Layout(
    annotations=[dict(x=0.02, y=0.98, xref="paper", yref="paper", text=f"Model R² = {r2_list[0]:.3f}",
                      showarrow=False, font=dict(size=18, color="gray"), align="left")],
    sliders=sliders,
    xaxis=dict(title="x", range=[-5.5, 5.5], tickfont=dict(size=14), fixedrange=True, gridwidth=1, zerolinewidth=1),
    yaxis=dict(title="y", range=[-3, 3], tickfont=dict(size=14), fixedrange=True, gridwidth=1, zerolinewidth=1),
    legend=dict(font=dict(size=14)),
    margin=dict(l=10, r=10, t=30, b=20),
)

# Create the figure
fig = go.Figure(data=data, layout=layout)
fig
```

As you probably expected, the R² increases as you increase the order of the regression model. However, it doesn't stop to do so after the 3rd order polynomial (which is the true shape of our data). The R² continues to increase until it hits 1 with a 29th order model. You can see that the model now goes through every single one of our data points. This did not happen by chance! A polynomial of degree 29 will perfectly interpolate our data, which consists of 30 data points. This is because a polynomial of degree n−1 has n coefficients, which can be uniquely determined to pass through n distinct points (assuming the x-values are distinct).

But what should you do with this information? Well, as the topic of this seminar is *statistical and machine learning*, we will be concerned with how to train models to make predictions for new, unseen data:

```{code-cell} ipython3
:tags: [remove-input]

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Fit Models on training data
np.random.seed(42)
x_all = np.linspace(-5, 5, 60)
indices_train = np.sort(np.random.choice(np.arange(60), size=30, replace=False))
x_train = x_all[indices_train]
y_train = (x_train**3 + np.random.normal(0, 15, size=x_train.shape)) / 50
df_train = pd.DataFrame({'x': x_train, 'y': y_train})

# Precompute polynomial coefficients fitted on training data for orders 1 to 30
training_coeffs = []
for order in range(1, 31):
    coeffs = np.polyfit(df_train['x'], df_train['y'], order)
    training_coeffs.append(coeffs)

#  Generate test data
np.random.seed(100)
x_all_test = np.linspace(-5, 5, 60)
indices_test = np.sort(np.random.choice(np.arange(60), size=30, replace=False))
x_test = x_all_test[indices_test]
y_test = (x_test**3 + np.random.normal(0, 15, size=x_test.shape)) / 50
df_test = pd.DataFrame({'x': x_test, 'y': y_test})

# Create a scatter trace for test data
test_scatter = go.Scatter(
    x=df_test['x'],
    y=df_test['y'],
    mode='markers',
    marker=dict(size=10, color='lightgrey', line=dict(color='gray', width=2)),
    name='Test Data'
)

# Compute Test Predictions and R² for Each Polynomial Order
regression_traces_test = []
r2_test_list = []
x_fit = np.linspace(-5, 5, 400)
for order in range(1, 31):
    coeffs = training_coeffs[order-1]  # model fitted on training data
    # Generate the regression curve using the training coefficients
    y_fit = np.polyval(coeffs, x_fit)
    trace = go.Scatter(
        x=x_fit,
        y=y_fit,
        mode='lines',
        name='Model',
        visible=False,
        line=dict(width=3, color='#4c72b0')
    )
    regression_traces_test.append(trace)
    
    # Compute test predictions for the test data
    y_pred_test = np.polyval(coeffs, df_test['x'])
    r2_test = 1 - np.sum((df_test['y'] - y_pred_test)**2) / np.sum((df_test['y'] - df_test['y'].mean())**2)
    r2_test_list.append(r2_test)

# Build Interactive Figure for Test Data
data_test = [test_scatter] + regression_traces_test
data_test[1]['visible'] = True

# Build slider steps: each step makes one regression curve visible and updates the annotation with the test R².
steps_test = []
for i in range(30):
    vis = [True] + [False] * 30
    vis[i + 1] = True
    step = dict(
        method="update",
        args=[{"visible": vis},
              {"annotations": [dict(x=0.02, y=0.98, xref="paper", yref="paper", 
                                    text=f"Test R² = {r2_test_list[i]:.3f}",
                                    showarrow=False, font=dict(size=18, color="gray"), align="left")]}],
        label=str(i + 1)
    )
    steps_test.append(step)

sliders_test = [dict(
    active=0,
    currentvalue={"prefix": "Order of polynomial: "},
    pad={"t": 30},
    steps=steps_test
)]

layout_test = go.Layout(
    annotations=[dict(x=0.02, y=0.98, xref="paper", yref="paper",
                      text=f"Test R² = {r2_test_list[0]:.3f}",
                      showarrow=False, font=dict(size=18, color="gray"), align="left")],
    sliders=sliders_test,
    xaxis=dict(title="x", range=[-5.5, 5.5], tickfont=dict(size=14), fixedrange=True, gridwidth=1, zerolinewidth=1),
    yaxis=dict(title="y", range=[-3, 3], tickfont=dict(size=14), fixedrange=True, gridwidth=1, zerolinewidth=1),
    legend=dict(font=dict(size=14)),
    margin=dict(l=10, r=10, t=30, b=20),
)

fig_test = go.Figure(data=data_test, layout=layout_test)
fig_test
```