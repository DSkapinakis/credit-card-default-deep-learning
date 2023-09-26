
<div align="center"> <img src="https://github.com/DSkapinakis/credit-card-default-deep-learning/assets/136902596/3e67cbe4-66bc-42a4-bc9f-508199897f61" width="500" height="500" alt="Image" justify-content: center> </div>


# A Comparative Analysis of Deep Learning Sequential Models for Temporal Feature Extraction in Credit Card Default Prediction

Developed as an MSc dissertation for the program "Business Analytics: Operational Research and Risk Analysis" at the Alliance Manchester Business School.

# Abstract
Three popular Deep Learning (DL) models (RNN, LSTM, CNN) were developed using various experimental setups, consisting of different final classification layers and input features. The purpose of this study was to discover the best performing DL model that extracts the dynamic characteristics of raw transactional data and is able to predict credit card default without the need for feature engineering. The LSTM model outperformed the other two, showing remarkable ability in identifying both defaulters and non-defaulters. This breakthrough, leveraging AI, has the potential to save financial institutions significant amounts of money by improving their credit risk assessment processes.

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

## Code structure
The folder _python code_ in this github repository contains 3 files, namely:
-  _EDA.ipynb_: jupyter notebook with all the exploratory data analysis (EDA) plots of the project
-  _hyperparameter_tuning.py_: python file including the whole tuning process for all 24 models developed in this study (architecture tuning + optimizer, learning rate, batch size tuning - see _elbow plots - tuning_ below)
-  _results_on_test_set_optimal_conf.py_: python file including the optimal configurations derived after tuning, and the final results on the test set

## Additional folders/files in this repository
-  _data_: folder containing the dataset
-  _elbow plots - tuning_: Excel files containing F1 scores for each architecture tested in the first tuning phase (architecture tuning). Those files are called in _hyperparameter_tuning.py_ to plot elbow curves and select the architecture that yields the highest F1 score with the lowest possible complexity. The selected optimal architectures were then tuned further (second tuning phase) regarding different optimizer, learning rates, batch sizes.
-  _technical_graphs_: pdf file containing some graphs/visualizations regarding the modelling pipeline and the different experimental setups used in this study. Also, visualizations are given to better understand how the 3 sequential DL models work. 

# Data
`default_of_credit_card_clients.xls`: dataset obtained from the UCI Machine Learning Repository (Yeh, 2016) and contains information about customers’ credit card payment history from a Taiwanese bank. Available in: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

<img src="https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/75c6a38d-06b9-4b1d-a38f-a5f8f814ca85" width="600" height="600" alt="Image Description">
 
# Research Design

The graphs below demonstrate the 8 stages in which the DL models were compared. The stages consist of the use of different final classification layers, namely a dense layer, logistic regression, random forest and XGBoost, and whether or not static features are included together with the temporal ones. The optimal Temporal Feature Extractor (TFE) will be the one which consistently demonstrates superior performance across the majority of stages. Each model undergone rigorous hyperparameter tuning, before the final assessment, regarding architecture, optimizer, learning rate and batch size.

![image](https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/a21fc75b-6071-4cc7-9f54-fb94a79e70ae)

![image](https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/11e62d74-2ba8-4c03-894a-2360e44d30ad)

# Results 

## Hyperparameter tuning - optimal configurations
The table presented below showcases the final configurations of 24 models spanning across 8 IF + FCL stages. Accompanying this information are the mean F1 scores and standard deviations derived from the 5-fold cross-validation procedure. It is noteworthy that, irrespective of the chosen input features, LSTM models, whether integrated with a dense layer or LR as the final classifier, consistently demonstrated superior performance in terms of mean F1 scores (the appendix mentioned is not available in this repository).

![image](https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/29e2bc51-93f9-4827-b064-c65350c8768d)

Below, a summarization of the optimal configurations occured after tuning for each DL model can be seen:
### RNN
<img src="https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/fa899fae-98c7-46c5-9b85-c57c4fab24ea" width="600" height="400" alt="Image">

### LSTM
<img src="https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/387d9f32-efc8-43aa-b541-786162c0a704" width="600" height="400" alt="LSTM Image">

### CNN
<img src="https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/049b7e55-c352-45cf-9e68-6c0c6aa21d37" width="600" height="400" alt="CNN Image">

## Final Classification results for all TFEs in each IF + FCL stage
As observed below, the RNN demonstrated its superior performance in 3 out of 8 stages (specifically, stages 3, 4, and 5), while the LSTM outperformed the others in 6 out of 8 stages (namely, stages 1, 2, 4, 5, 6, and 7). The CNN exhibited its strength in 2 out of 8 stages (specifically, stages 6 and 8). It's worth noting that the highest F1 scores achieved (0.481, 0.482, 0.488) were consistently associated with LSTM as the TFE.

Additionally, it's important to highlight that models employing RF and XGB (in stages 3, 4, 7, and 8) as the final classification layer tended to exhibit signs of overfitting. However, this pattern was consistent across all three TFEs in these stages, suggesting that it did not adversely impact the comparative analysis between them.

In conclusion, irrespective of the input features or the final classification layer chosen, LSTM consistently emerged as a top-performing model in the majority of cases, demonstrating its robustness in our final comparison.

![image](https://github.com/DSkapinakis/credit-card-default-prediction-ml/assets/136902596/3c503f07-42aa-4074-8510-4b89a2541082)



