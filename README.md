Parkinsons Classification
=========================

As an exercise in using the [R](https://www.r-project.org/about.html) statistical programming language, I was interested in performing exploratory data analysis and building a binary classification model for the [Parkinsons Data Set.](https://archive.ics.uci.edu/ml/datasets/parkinsons) The dataset, publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) consists of predictor variables (X) which are various biomedical voice measurements taken from patient audio recordings, and annotations (Y) that identify the patients as being healthy or having Parkinson’s. 

The first part of the project involved investigating the relationship between potential predictor variables through analysis of the following graphical plots:
- Kernel Density Plots
- Correlation Matrices
- Boxplots

The ["caret"](https://cran.r-project.org/web/packages/caret/vignettes/caret.html) package, a streamlined interface for pre-processing data and implementing classification algorithms, was then used to test several algorithms in search for the "best" classification model for the problem at hand. Selection of the "best" model was based on Repeated K-Fold Cross-Validation and several metrics, most notably the AUC score (Area under the ROC curve), accuracy, sensitivity, and specificity. The following models were tested:
- Logistic Regression (Single Predictor)
- Logistic Regression (Multiple Predictors)
- Naive Bayes
- Regularized Logistic Regression
- Boosted Logistic Regression
- Random Forest

Random Forest was identified as the "best" model and its hyperparameters were further tuned via grid search to achieve more optimal performance of the algorithm. The final AUC score was **0.9884.**
