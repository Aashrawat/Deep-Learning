# LEARNING ANN

Iris species classification in `ANNPrj.ipynb`. A perceptron is the baseline. A small multilayer network is the main model.

## Symbols

| Symbol | Name | Meaning |
| --- | --- | --- |
| $x$ | feature vector | One flower’s four measurements. |
| $y$ | label | True species, stored as an integer $0$, $1$, or $2$. |
| $\hat{y}$ | prediction | Species the model chooses. |
| $z$ | score | Weighted sum $w^{\top} x + b$ before an activation. |
| $w, b$ | weights, bias | Numbers the model learns. |
| $a$ | activation | The value a layer passes to the next layer. |
| $\hat{p}$ | probability | Model’s distribution over the three species. |
| $J$ | cost | How wrong those probabilities are. |
| $m$ | sample size | Number of training flowers. |

## What an artificial neural network is

An **artificial neural network** stacks simple units. Each unit takes a vector $x$, computes a score, and passes that score through an **activation**:

$$
z = w^{\top} x + b, \qquad a = g(z)
$$

$w$ says how much each input matters. $b$ shifts the score. $g$ is the activation. One unit is a **neuron**. A **layer** is many neurons that all see the same input. A **multilayer** network feeds one layer’s outputs into the next.

Training changes $w$ and $b$ so the cost $J$ gets smaller. The notebook uses **gradient descent**: each step moves the weights a little in the direction that reduces $J$. Keras does that update. Adam is the optimizer in the multilayer model; it keeps a running average of the gradient so each weight can take its own step size.

## Perceptron

A **perceptron** is one linear unit. For two classes it predicts from the sign of the score:

$$
\hat{y} =
\begin{cases}
1 & \text{if } w^{\top} x + b \ge 0 \\
0 & \text{otherwise}
\end{cases}
$$

If the prediction is wrong, the weights move toward the correct side of that line. If it is right, they stay put. The boundary is a straight cut through the features, so a perceptron only separates classes that a hyperplane can split.

Iris has three species. scikit-learn’s `Perceptron` learns one weight vector per class and predicts the class with the largest score. That is still a linear rule. It cannot bend the boundary around overlapping classes.

## Hidden layers and ReLU

A multilayer network puts **hidden layers** between the inputs and the class scores. This notebook uses two. The first has 16 units, the second has 8. Both use **ReLU**:

$$
\mathrm{ReLU}(z) = \max(0, z)
$$

Negative scores become $0$. Positive scores pass through unchanged. Stacking ReLU layers lets the model combine the four measurements into features a single line cannot express. Petal length and petal width overlap for versicolor and virginica; the extra layers are what separate those two after setosa is already easy.

The forward pass in `ANNPrj.ipynb` is:

$$
a^{[1]} = \mathrm{ReLU}\!\left(W^{[1]} x + b^{[1]}\right)
$$

$$
a^{[2]} = \mathrm{ReLU}\!\left(W^{[2]} a^{[1]} + b^{[2]}\right)
$$

$$
z^{[3]} = W^{[3]} a^{[2]} + b^{[3]}
$$

$W^{[1]}$ is $16 \times 4$, $W^{[2]}$ is $8 \times 16$, and $W^{[3]}$ is $3 \times 8$.

## Softmax and the cost

The last layer has one score per species. **Softmax** turns those three scores into probabilities that add to $1$:

$$
\hat{p}_c = \frac{e^{z_c}}{e^{z_0} + e^{z_1} + e^{z_2}}
$$

The predicted species is the class with the largest $\hat{p}_c$.

Labels are one-hot with `to_categorical`: setosa is $(1,0,0)$, versicolor is $(0,1,0)$, virginica is $(0,0,1)$. The cost is **categorical cross-entropy**:

$$
J = -\sum_{c} y_c \log \hat{p}_c
$$

Only the true class contributes. Predicting a small probability for that class makes $J$ large. `categorical_crossentropy` is that cost, and `accuracy` is the fraction of flowers whose $\hat{y}$ matches $y$.

## The project

`ANNPrj.ipynb` classifies the iris table from `sns.load_dataset("iris")`. There are 150 flowers and no missing values. Each species has 50 rows.

| Column | Role |
| --- | --- |
| `sepal_length`, `sepal_width` | Features, in centimeters. |
| `petal_length`, `petal_width` | Features, in centimeters. |
| `species` | Label: setosa, versicolor, or virginica. |

A pair plot colored by species is drawn before any model is fit. Setosa sits apart on the petal measurements. Versicolor and virginica overlap.

`LabelEncoder` maps the names in alphabetical order: setosa $\to 0$, versicolor $\to 1$, virginica $\to 2$. A stratified $80/20$ split (`random_state=42`) keeps the class balance and leaves 120 training rows and 30 test rows. `StandardScaler` subtracts the mean and divides by the standard deviation so a column with a wider range does not dominate the weighted sum.

The notebook calls `fit_transform` on the training rows and again on the test rows. The test scaler therefore uses the test set’s own mean and variance.

### Perceptron baseline

`Perceptron(max_iter=1000, random_state=42)` reaches **0.90** accuracy on the 30 test rows.

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| setosa (0) | 1.00 | 1.00 | 1.00 | 10 |
| versicolor (1) | 1.00 | 0.70 | 0.82 | 10 |
| virginica (2) | 0.77 | 1.00 | 0.87 | 10 |

Setosa is separated cleanly. Three versicolor flowers are predicted as virginica.

### Multilayer network

| Layer | Units | Activation |
| --- | ---: | --- |
| Hidden | 16 | ReLU |
| Hidden | 8 | ReLU |
| Output | 3 | Softmax |

The model is compiled with Adam and categorical cross-entropy. Training runs for 100 epochs with batch size 8. `validation_split=0.2` holds out the last 24 of the 120 training rows and scores them at the end of each epoch.

On the saved run, epoch 100 is training accuracy **0.979** and validation accuracy **1.00**. `model.evaluate` on the test set prints **Test Accuracy: 0.967**. The last cell plots training accuracy and validation accuracy across the 100 epochs.

## Files

| File | What it shows |
| --- | --- |
| `ANNPrj.ipynb` | Load iris, plot the species, scale, fit a perceptron, then train the 16–8–3 network and report test accuracy. |

Open the notebook on GitHub or in [Colab](https://colab.research.google.com/github/Aashrawat/Deep-Learning/blob/main/ANN/ANNPrj.ipynb).

## Pipeline in the notebook

1. Load `sns.load_dataset("iris")` and draw `sns.pairplot` colored by `species`.
2. $X$ is the four measurement columns. $y$ is `species`.
3. `LabelEncoder` maps species names to $0$, $1$, $2$.
4. Stratified $80/20$ split (`random_state=42`).
5. `StandardScaler` on the training rows, then again on the test rows.
6. Fit `Perceptron` and print accuracy plus `classification_report`.
7. One-hot encode the labels with `to_categorical`.
8. Build `Dense(16, relu)` $\to$ `Dense(8, relu)` $\to$ `Dense(3, softmax)`.
9. Compile with Adam and `categorical_crossentropy`. Fit for 100 epochs, batch size 8, `validation_split=0.2`.
10. Evaluate on the test set and plot training accuracy against validation accuracy.
