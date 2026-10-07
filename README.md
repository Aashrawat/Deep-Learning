# Deep-Learning

Notebooks for learning neural networks with Keras and TensorFlow. The artificial-neural-network work is in the notebooks below. The CNN and RNN folders are reserved for later.

## Contents

| Path | Topic |
| --- | --- |
| [`Basic.ipynb`](Basic.ipynb) | Binary classification on a 16-row plant-watering example |
| [`ANN/ANNPrj.ipynb`](ANN/ANNPrj.ipynb) | Iris species: a perceptron baseline, then a small multilayer network |
| [`CNN/`](CNN/) | Convolutional networks (notebook not added yet) |
| [`RNN/`](RNN/) | Recurrent networks (notebook not added yet) |

## Run the notebooks

Both notebooks open in Google Colab:

- [Basic.ipynb](https://colab.research.google.com/github/Aashrawat/Deep-Learning/blob/main/Basic.ipynb)
- [ANN/ANNPrj.ipynb](https://colab.research.google.com/github/Aashrawat/Deep-Learning/blob/main/ANN/ANNPrj.ipynb)

To run them locally with Python 3:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
jupyter notebook
```

`Basic.ipynb` uses NumPy, pandas, scikit-learn, and TensorFlow. The iris notebook also uses Matplotlib and Seaborn. `sns.load_dataset('iris')` downloads the iris table the first time it runs.

## Basic.ipynb

A one-hidden-layer network that predicts whether a plant needs water.

The table is 16 handwritten rows. The features are soil moisture, temperature in °C, and hours of sunlight. The label `needs_water` is 0 or 1.

1. Each feature is min-max scaled into [0, 1].
2. A stratified 75/25 split (`random_state=42`) leaves 12 training rows and 4 test rows.
3. The Keras `Sequential` model is 3 inputs, `Dense(8, relu)`, then `Dense(1, sigmoid)`.
4. Training uses SGD and binary cross-entropy for 100 epochs with batch size 1.

The saved run finishes at training accuracy 1.00 and validation accuracy 1.00. The validation set is those four test rows, so the score only shows that this small table is separable.

## ANN/ANNPrj.ipynb

Iris species classification. Seaborn loads 150 flowers, four measurements (sepal length, sepal width, petal length, petal width), and three species: setosa, versicolor, and virginica. The notebook draws a pair plot before fitting anything.

`LabelEncoder` maps the species names to integers 0, 1, and 2 in alphabetical order (setosa, versicolor, virginica). A stratified 80/20 split (`random_state=42`) gives 120 training rows and 30 test rows. Features go through `StandardScaler`.

### Perceptron baseline

scikit-learn’s `Perceptron` (`max_iter=1000`) reaches **0.90** accuracy on the 30 test rows.

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

The model is compiled with Adam and categorical cross-entropy. Labels are one-hot encoded with `to_categorical`. Training runs for 100 epochs at batch size 8, and `validation_split=0.2` holds out 24 of the 120 training rows for validation.

On the saved run, epoch 100 is about **0.98** training accuracy and **1.00** validation accuracy. `model.evaluate` on the test set prints **Test Accuracy: 0.967**.

The test rows are scaled with `scaler.fit_transform`, which fits a new mean and variance on the test set. `scaler.transform` reuses the scaler that was fitted on the training rows.

## CNN and RNN

[`CNN/`](CNN/) and [`RNN/`](RNN/) only contain placeholder README files. Notebooks for convolutions and recurrent nets are not in the repo yet.
