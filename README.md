# Ensemble-Based Machine Learning Approach for Accurate and Cost-Effective Early Detection of Harmful Algal Blooms

This repository is an official implementation of [Ensemble-Based Machine Learning Approach for Accurate and Cost-Effective Early Detection of Harmful Algal Blooms](link).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Processing Data

To process the data, run this command:

```process
python preprocess.py
```

## Training

To train the each model (random forests, gradient boosting model, neural network) in the paper, run this command once:

```train
python epoch_training.py
```

## Evaluation

To evaluate the model loss, run:

```eval
python loss_RMSE.py
```

To evaluate the model's residuals, run:

```eval
python residuals.py
```

To evaluate the model overall cross validation, run:

```eval
python cv_plot.py
```

## Pre-trained Models

You can download pretrained models here:

- [Ensemble model](link) trained on Dataset.xlsx using tuned parameters. 


## Results

Our model achieves the following performance on: