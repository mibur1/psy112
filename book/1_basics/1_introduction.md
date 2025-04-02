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

# <i class="fa-solid fa-code"></i> What is Machine Learning?


```{code-cell} ipython3
:tags: [remove-input]

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go

warnings.filterwarnings("ignore", message=".*Polyfit may be poorly conditioned.*")
# Generate sample data
np.random.seed(41)
x = np.linspace(-5, 5, 30)
y = (x**3 + np.random.normal(0, 15, size=x.shape)) / 50
df = pd.DataFrame({'x': x, 'y': y})

# Scatter trace for the raw data
scatter = go.Scatter(x=df['x'], y=df['y'],
                     mode='markers',
                     marker=dict(color='black'),
                     name='Data')

# Generate regression curves for polynomial orders 1 through 50
regression_traces = []
x_fit = np.linspace(-5, 5, 400)
for order in range(1, 31):
    coeffs = np.polyfit(df['x'], df['y'], order)
    y_fit = np.polyval(coeffs, x_fit)
    trace = go.Scatter(x=x_fit, y=y_fit,
                       mode='lines',
                       name=f'Regression',
                       visible=False)  # hide initially
    regression_traces.append(trace)

# Combine data: always show the scatter trace, and one regression trace at a time
data = [scatter] + regression_traces
data[1]['visible'] = True # Set the first regression trace to be visible

# Create slider steps
steps = []
for i in range(0,30):
    vis = [True] + [False] * 50
    vis[i + 1] = True
    step = dict(
        method="update",
        args=[{"visible": vis}],
        label=str(i + 1)
    )
    steps.append(step)

# Define the slider
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Polynomial Order: "},
    pad={"t": 30},
    steps=steps
)]

# Define the layout
layout = go.Layout(
    sliders=sliders,
    xaxis=dict(range=[-5, 5]),
    yaxis=dict(range=[-3, 3])
)

# Create the figure
fig = go.Figure(data=data, layout=layout)
fig
```

Have you ever wondered how Netflix always seems to know exactly what movie or TV show to recommend next? The secret lies in machine learning. By analyzing your viewing history, ratings, and even the habits of viewers with similar tastes, Netflix's algorithms detect patterns and predict which content you might enjoy. It’s not just random suggestions — it's a sophisticated system that processes vast amounts of data to tailor your recommendations, ensuring that your next binge-watching session feels perfectly curated just for you.

But what is machine learning? Machine learning is a branch of artificial intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed. Rather than relying on fixed instructions, machine learning algorithms analyze vast amounts of data to identify meaningful patterns and predict outcomes.


## Types of learning

Machine learning can be categorized into three main types based on the way the model learns from data:

- **Supervised learning:**    Uses labeled data to train models, making predictions or decisions based on examples (e.g., predicting mental health scores based on questionnaire responses).
- **Unsupervised learning:**  Finds patterns or structures in unlabeled data (e.g., clustering individuals based on similar traits or behaviors).
- **Reinforcement learning:** Involves an agent interacting with an environment, learning through trial and error with feedback from its actions.

In this course, we will focus on the first two types of learning. If you are interested in reinforcement learning, you can find a nice introduction [here](https://mlu-explain.github.io/reinforcement-learning/).

```{figure} figures/ML.drawio.png
:name: ML
:alt: Types of learning
:align: center

Three main types of machine learning: Supervised, unsupervised, and reinforcement learning.
```


## Classes of algorithms

In addition to the type of learning, algorithms can also be categorized based on the type of data that needs to be predicted:

| Type of data                  | Type of learning                                  | Type of learning                                       |
|-------------------------------|---------------------------------------------------|--------------------------------------------------------|
|                               | **Supervised learning** <br /> (labeled outcomes) | **Unsupervised learning** <br /> (no labeled outcomes) |
| **Qualitative (categorical)** | Classification                                    | Clustering                                             |
| **Quantitative (numerical)**  | Regression                                        | Dimensionality reduction <br /> Clustering             |


## Assessing performance

Assessing the performance of a machine learning model is crucial to ensure its reliability and applicability. Important evaluation metrics for this are:

- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values (regression tasks).
- **Accuracy**: Describes the proportion of correctly predicted outcomes (classification tasks).

There are many more performance metrics (which you might already know), such as the Root Mean Squared Error (RMSE), R-Squared (R²), Precision, or a confusion matrix. You will get to know some of them in the following weeks.

Additionally, techniques like *cross-validation* help estimate a model’s performance more reliably. Cross-validation splits the dataset into training and testing subsets, reducing the risk of overfitting and proving a better assessment of how the model will perform on unseen data. We will explore this topic in the [](3_resampling) session.
