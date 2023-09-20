<img src="https://github.com/DSkapinakis/credit-card-default-deep-learning/assets/136902596/3e67cbe4-66bc-42a4-bc9f-508199897f61" width="420" height="480" alt="Image">

<img src="https://github.com/DSkapinakis/credit-card-default-deep-learning/assets/136902596/61d5900c-e21d-4354-9078-33183f2a331a" width="570" height="480" alt="Image">


# A Comparative Analysis of Deep Learning Sequential Models for Temporal Feature Extraction in Credit Card Default Prediction


Developed as an MSc dissertation for the program "Business Analytics: Operational Research and Risk Analysis" at the Alliance Manchester Business School.


# Abstract
The surge in credit card transactional data compels financial institutions to develop accurate credit scoring models capable of identifying potential defaults. Proper feature engineering of data is essential for such models to perform well, a process that is often time-consuming and requires a high level of human expertise. Deep Learning (DL) has received significant attention for its ability to automatically extract high-level representations from raw data, thus reducing human intervention and, by extension, bias. Due to the temporal nature of transactional data, researchers have developed several sequential DL models capable of extracting inherent temporal dependencies and accurately predicting default cases. Most applications concern the ‘Taiwan’ dataset, a credit card dataset that includes both static and temporal customer features. Despite the numerous studies, each researcher employs a single DL model for temporal feature extraction and adopts a unique approach regarding the modelling pipeline, the inclusion or not of static features, and the final classification layer. This variation in approaches makes the different studies incomparable, highlighting the absence of a comprehensive assessment of the sequential DL model that yields the highest classification scores. This study aims to cover this gap by assessing three popular sequential DL models ‒ Recurrent Neural Network (RNN), Long-Short Term Memory Network (LSTM), and Convolutional Neural Network (CNN) ‒ across eight stages, using a consistent modelling pipeline. Each stage combines different final classification layers ‒ Dense, Logistic Regression, Random Forest, XGBoost ‒ with inclusion or not of static features. After performing hyperparameter tuning for each unique sequential DL model, F1 score was the main metric utilized for the final evaluations, revealing LSTM’s superiority in 6 out of 8 stages. RNN came second with 3 appearances as a top performer and CNN followed with 2. This study will guide financial institutions in adopting advanced DL approaches for credit scoring.

**Keywords:** Machine Learning, Deep Learning, Temporal Feature Extraction, Credit Card Default 

# Installation and Setup

## Codes and Resources Used
- **IDE Used:**  VSCode
- **Python Version:** 3.10.9

## Python Packages Used

- **General Purpose:** `copy`, `os`, `random`
- **Data Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `seaborn`, `matplotlib` 
- **Machine Learning Models - Preprocessing:** `scikit-learn`, `xgboost`
- **Deep Learning Models:** `tensorflow`, `keras`

# Data
`default_of_credit_card_clients.xls`: dataset obtained from the UCI Machine Learning Repository (Yeh, 2016) and contains information about customers’ credit card payment history from a Taiwanese bank. Available in: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

<img src="https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/75c6a38d-06b9-4b1d-a38f-a5f8f814ca85" width="600" height="600" alt="Image Description">
 
# Research Design
To determine the most powerful DL model, among the most popular ones, for temporal feature extraction in the Taiwan dataset, the three investigated models will be assessed across eight stages, ensuring a comprehensive and robust evaluation. The stages were derived by utilizing either temporal only or static and temporal input features (IF), and by performing the final classification with either a dense layer or a conventional ML model. The temporal feature extractor (TFE) that consistently demonstrates superior performance across the majority of stages will be chosen as the optimal one. Following the exploratory data analysis (EDA), the final results will be produced for the optimized versions (hyperparameter tuning) of all 24 models (3x8 stages). The figures below depict the research design pipeline and the composition of the eight IF + FCL (Final Classification Layer) stages.

![image](https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/a21fc75b-6071-4cc7-9f54-fb94a79e70ae)

![image](https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/11e62d74-2ba8-4c03-894a-2360e44d30ad)

# Code structure
The folder _python code_ in this github repository contains 3 files, namely:
-  _EDA.ipynb_: jupyter notebook with all the exploratory data analysis (EDA) plots of the project
-  _hyperparameter_tuning.py_: python file including the whole tuning process for all 24 models developed in this study (architecture tuning + optimizer, learning rate, batch size tuning - see _elbow plots - tuning_ below)
-  _results_on_test_set_optimal_conf.py_: python file including the optimal configurations derived after tuning, and the final results on the test set

# Additional folders/files in this repository
-  _data_: folder containing the dataset
-  _elbow plots - tuning_: Excel files containing F1 scores for each architecture tested in the first tuning phase (architecture tuning). Those files are called in _hyperparameter_tuning.py_ to plot elbow curves and select the     architecture that yields the highest F1 score with the lowest possible complexity. The selected optimal architectures were then tuned further (second tuning phase) regarding different optimizer, learning rates, batch sizes.
-  _technical_graphs_: pdf file containing some graphs/visualizations regarding the modelling pipeline and the different experimental setups used in this study. Also, visualizations are given to better understand how the 3         sequential DL models work. 

# Results 

## Hyperparameter tuning - optimal configurations
The table below demonstrates the final configurations of the 24 models across the 8 IF + FCL stages, alongside the mean F1 scores and standard deviations obtained through the 5-fold cross validation. Regardless of the input features, LSTM models with a dense layer or with LR as a final classifier, achieved the highest mean F1 scores (the appendix mentioned is only available in the dissertation file).

![image](https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/29e2bc51-93f9-4827-b064-c65350c8768d)

Below, a summarization of the optimal configurations occured after tuning for each DL model can be seen:
### RNN
<img src="https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/fa899fae-98c7-46c5-9b85-c57c4fab24ea" width="600" height="400" alt="Image">

### LSTM
<img src="https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/387d9f32-efc8-43aa-b541-786162c0a704" width="600" height="400" alt="LSTM Image">

### CNN
<img src="https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/049b7e55-c352-45cf-9e68-6c0c6aa21d37" width="600" height="400" alt="CNN Image">

## Final Classification results for all TFEs in each IF + FCL stage
As seen below, the RNN demonstrated its superiority in 3 out of 8 stages (stages 3, 4, 5), the LSTM in 6 out of 8 (stages 1, 2, 4, 5, 6, 7) and the CNN in 2 out of 8 stages (stages 6, 8). As an additional observation, the highest F1 scores obtained (0.481, 0.482, 0.488) across all 24 configurations, encompass LSTM as the TFE. Lastly, it should be noted that the models with RF and XGB (stages 3, 4, 7, 8) as FCL clearly overfit. However, this behavior is consistent across all 3 TFEs in these stages, implying that this behavior does not negatively impact the comparison between them.
Regardless of the input features or the final classification layer, LSTM keeps on being a top performer in the majority of cases, providing robustness in the final comparison. 

![image](https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/3c503f07-42aa-4074-8510-4b89a2541082)



