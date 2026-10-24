This repository contains all the supplemental files and scripts for the Nature Communications paper 'Predicting and Improving Complex Beer Flavor through Machine Learning'.

Init_modeling.py

Helper script that contains some functions for model performance evaluation and storage.



Jupyter Notebook for training machine learning models.ipynb

Jupyter notebook containing the code to train the 10 different machine learning models of the research article, including hyperparameter optimization and model performance evaluation.

The 10 models included are:

Linear regression
Lasso regression
Partial least squares regression
Random Forest regression
Extra-trees regression
Adaboosted regression
Gradient boosting regression
XGBoosting regression
Support vector regression
Multi-layered perception



Machine learning models transcript.py

A direct copy of the Jupyter Notebook, as a regular Python script.



Main figure generator.Rmd

R Markdown file that generates the figures from the paper's main text.



Sensory data quality control - ANOVA on repeated samples.Rmd

R Markdown file that performs ANOVA on the beer samples that were repeated in multiple sessions with the trained panel.



Supplemental Figure generator.Rmd

R Markdown file that generates the supplemental figures of the paper.



Supplemental Files and Figure source files.xlsx

Excel file containing all the supplemental files and tables, and the raw data underlying every figure (main and supplemental) in the manuscript.



RateBeer_Appreciation_Gradient_Boost_Regressor.pkl

Pickled version of the best-performing model for predicting RateBeer Appreciation from chemical composition (Python version 3.9.16).



RateBeer_Full_Model_Gradient_Boost_Regressor.pkl

Pickled version of the best-performing model for predicting all RateBeer scores, from beer chemical composition (Python version 3.9.16).



SHAP analysis.ipynb

Jupyter Notebook that performs SHAP analysis on the models. It also extracts the SHAP values in a CSV file to make a customized SHAP plot in R with the "Main Figure Generator.Rmd" script.



Partial Dependence Plots with SHAP.ipynb

Jupyter Notebook that generates Partial Dependence Plots of your model for compounds of your choice.









SYSTEM REQUIREMENTS

Any operating system capable of running Python v3.9.16 and R v4.2.2

Python v3.9.16 packages:

anaconda	2023.07
conda	23.5.0
joblib	1.2.0
numpy	1.24.3
pandas	1.5.3
scikit-learn	1.2.2
seaborn	0.12.2
shap	0.41.0
xgboost	1.7.3


R v4.2.2 packages:
factoextra	1.0.7
forcats	1.0.0
ggbeeswarm	0.7.1
ggpmisc	0.5.2
ggpubr	0.6.0
ggsignif	0.6.4
ggtext	0.1.2
grid	4.2.2
gridExtra	2.3
Hmisc	4.8.0
naniar	1.0.0
psycho	0.6.1
RColorBrewer	1.1.3
shapes	1.2.7
stats	4.2.2
svglite	2.1.1
tidyquant	1.0.6
tidyverse	2.0.0



INSTALLATION GUIDE

Download and install Python from Python or Anaconda.
Download and install R from CRAN.

Packages are installed with the built-in package installers.



DEMO / INSTRUCTIONS FOR USE

All available data is included in the file "Supplemental Files and Figure source files.xlsx".
Open the worksheet of your choise and save this as a .CSV file.

All scripts refer to the corresponding worksheet names whenever the data is required.

The expected output is identical to the figures, tables and supplemental files presented in the manuscript.

