# Digit Classifier app

View the deployed application here: https://xgb-digit-classifier.herokuapp.com/

This repo demonstrates how to build a classifier using 
* mnist dataset
* ensemble classifiers (random forest, xgboost)

Important files:
* `initial-model-training.ipynb`: preprocess the dataset, train the models
* `optimization`: this folder contains alternative versions of the trained models, using gridsearch
* `rf_grid_model.tar.gz`: this is a compressed file containing the `pkl` file because the original was too large for github.


I was inspired by these other apps:
* http://benjaminlu-ds-digits.herokuapp.com/
* https://benjaminlu-ds-digits-v2.herokuapp.com/
* https://quickdraw-10-classification.herokuapp.com/
* https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-canvas-ocr
