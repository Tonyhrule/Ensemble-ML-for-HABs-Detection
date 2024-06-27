# Ensemble-Based Machine Learning Approach for Accurate and Cost-Effective Early Detection of Harmful Algal Blooms

This repository contains the official implementation of the research paper Ensemble-Based Machine Learning Approach for Accurate and Cost-Effective Early Detection of Harmful Algal Blooms. The approach utilizes ensemble learning techniques and multi-agent frameworks to enhance the early detection of harmful algal blooms (HABs).

## Overview

Harmful Algal Blooms (HABs) pose significant ecological and economic threats, costing the U.S. $50 million annually and causing public health issues and water quality degradation. Conventional detection methods, such as manual sampling, analysis, satellite monitoring, and sensing, are time-consuming, costly, and lack real-time monitoring capabilities. This study proposes an ensemble-based machine learning approach to predict corrected Chlorophyll-a concentration, a key indicator of HABs. Initial testing of our ensemble model revealed poor performance in predicting higher Chlorophyll-a concentrations. To address this, we developed a multi-agent collaborative framework utilizing large language models (LLMs) for data augmentation, generating synthetic data to improve model accuracy. By training three different types of models on historical and synthetic data and averaging their predictions, we improved the accuracy of our ensemble approach, achieving a 4.77% reduction in prediction error compared to the ensemble model trained solely on the original dataset. Moreover, our final model demonstrated a promising 66.10% improvement in RMSE compared to conventional models using satellite data. This approach offers a scalable, cost-effective solution for early HAB detection, enhancing AI-driven environmental monitoring and prediction systems.

## Requirements

To install the required dependencies, run:

```setup
pip install -r requirements.txt
```

## Processing Data

Before processing the data, ensure that the [output folder](https://github.com/Tonyhrule/Ensemble-ML-for-HABs-Detection/tree/main/output) and its contents are deleted if they exist, as the `preprocess.py` script will generate new files and the output folder.

To process the data, run this command:

```process
python preprocess.py
```

## Training

This project includes training four models: Random Forests, Gradient Boosting, Neural Networks, and a Stacked/Ensemble model. To train all models at once, run:

```train
python train.py
```

## Evaluation

We provide multiple scripts to evaluate the models' performance on different metrics:

To evaluate the Root Mean Squared Error (RMSE) of the models, run:

```eval
python loss_RMSE.py
```

To evaluate the residuals of the models' predictions, run:

```eval
python residuals.py
```

To evaluate the models' percent error, run:

```eval
python percent_error.py
```

## Pre-trained Models

You can download pre-trained models from the following link:

- [Google Drink]([link](https://drive.google.com/drive/folders/1Adxt7VVraiiV6TuSErsl2ydmaW2flUJU?usp=sharing))

## Results



## Web-Application Demo

We have developed a simple web application to demonstrate the functionality and use of our model for HAB prediction/detection. You can access it here: [Web-app demo](https://predicthabs.streamlit.app/).

The web application is built using Streamlit. The code for the app is in the following folder: [Streamlit App Code](https://github.com/Tonyhrule/Ensemble-ML-for-HABs-Detection/blob/main/streamlit_app.py).

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/your-username/Ensemble-ML-for-HABs-Detection.git`
3. Create a new branch: `git checkout -b feature-or-bugfix-name`
4. Make your changes and commit: `git commit -m "Description of changes"`
5. Push to your fork: `git push origin feature-or-bugfix-name`
6. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Tonyhrule/Ensemble-ML-for-HABs-Detection/blob/main/LICENSE) file for details.
