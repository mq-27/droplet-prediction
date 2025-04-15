# Microfluidic droplet size prediction
Interpretable tree-based ensemble models for microfluidic droplet size prediction
## Description
The prediction of the droplet size depends on complex and multiple factors, including microchannel geometries, experimental systems, operation conditions and so on.  An data and mechanism dual driven framework for exploring microfluidic droplet generation is developed. Please refer to our work "Modeling and analysis of droplet generation in microchannels using interpretable machine learning methods" for additional details.

## Installation
1.Create conda environment<br>
````
conda create -n name python=3.8.16
conda activate name
````
2.Install the following python package:
````
conda install hyperopt=0.2.7
conda install joblib=1.2.0
conda install matplotlib=3.7.1
conda install numpy=1.24.3
conda install pandas=2.0.2
conda install scikit-learn=1.2.2
conda install seaborn=0.12.2
conda install shap=0.44.0
conda install xgboost=1.7.6
````

## Details:
### Folders
**main**: Core code of the project <br>
**demo**: Provide data from "Droplet Generation in Micro-sieve Dispersion Device" as a demo to validate the models<br>
**fig**: Overall workflow of the project<br>
### Code in the main folder
**data pre-processing.ipynb**: Data cleaning and feature engineering, exploration of the dataset, visualization of feature distribution<br>
**DT.ipynb**: Decision Tree models optimized with Bayesian<br>
**RF.ipynb**: Random Forest models optimized with Bayesian <br>
**GBDT.ipynb**: Gradient Boosting Decision Tree models optimized with Bayesian<br>
**stacking.ipynb**: Stacking models<br>
**shap.ipynb**: Visual Interpretation of Machine Learning Models<br>
**function.py**: Evaluation metrics, cross_regressor, prediction plot <br>

## Authors

| **AUTHORS** |Mengqi Liu            |
|-------------|----------------------|
| **VERSION** | 1.0 / May,2024                               |
| **EMAILS**  | liumq22@mails.tsinghua.edu.cn                         |

## Publication
https://doi.org/10.1016/j.cej.2025.161972

## Attribution
This work is under MIT License. Please, acknowledge use of this work with the appropiate citation to the repository and research article.
