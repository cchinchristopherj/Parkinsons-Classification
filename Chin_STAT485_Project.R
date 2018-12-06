#' Project
#'
#' 1) 
#' The dataset that will be used for this project is the Parkinsons Data Set from the UCI Machine Learning Repository. The dataset consists of predictor variables (X) which are various biomedical voice measurements taken from patient audio recordings, and annotations (Y) that identify the patients as being healthy or having Parkinson’s. The predictor variables include the average vocal fundamental frequency in the recording, and measures of variation in frequency and amplitude over the course of the recording. The data was taken from a study of 31 patients, 23 of whom had Parkinson’s, with each person contributing about 6 recordings to the dataset. Each row in the .csv file corresponds to all the predictor variables associated with one recording and one column called “status” is a binary variable indicating the patient as having Parkinson’s (1) or being healthy (0). The .csv file had no empty rows or cells, and did not require additional cleaning prior to being imported. However, two rows of comments prefaced by “#” were added above the data to clarify the contents of the file. 
#' First load in the required libraries
library(caret)
library(corrplot)
library(DataExplorer)
library(naivebayes)
library(ranger)
library(stepPlr)
library(pROC)
set.seed(1237)
#' Use read.csv() to load in the 'parkinsons.csv' dataset.
my.data = read.csv('Data/parkinsons.csv',comm='#')
#' The "names" column contains the id numbers associated with each patient and is unnecessary for the subsequent analysis. Remove the column from the data frame.
my.data = my.data[,-1]
summary(my.data)
#' The "status" column is a binary indicator variable that is interpreted by the data frame as quantiative. Ensure that it is a numeric vector by using as.numeric().
my.data$status = as.numeric(my.data$status)
names(my.data)

#' Exploration
#' The first visualization I will use to perform exploratory data analysis is a correlation matrix. Concretely "corrplot" from the corrplot package will be used to graphically depict the matrix of correlation coefficients between each combination of predictor variables. 
corrplot(cor(my.data),method='color')
cor(my.data)
#' There appear to be some strong correlations between the predictors, particularly between predictors related to "Shimmer." Interestingly, however, no variable appears to be strongly correlated with the response variable "status." Only "spread1" (and "spread1" and "spread2") are correlated slightly with "status." A model created with solely "spread1" as a predictor could be useful, but due to the relatively low correlation coefficient, it would be more practical to include multiple predictors. 
#' The second visualization I will use is plots of the distribution of each predictor variable. Kernel density plots will be used instead of histograms, the results of which depend on the number of bins used.
op = par(mfrow=c(4,3),mar=c(4,4,4,4))
plot(density(my.data$MDVP.Fo.Hz.),main="MDVP.Fo.Hz.")
plot(density(my.data$MDVP.Fhi.Hz.),main="MDVP.Fhi.Hz.")
plot(density(my.data$MDVP.Flo.Hz.),main="MDVP.Flo.Hz.")
plot(density(my.data$MDVP.Jitter...),main="MDVP.Jitter")
plot(density(my.data$MDVP.Shimmer),main="MDVP.Shimmer")
plot(density(my.data$NHR),main="NHR")
plot(density(my.data$HNR),main="HNR")
plot(density(my.data$RPDE),main="RPDE")
plot(density(my.data$DFA),main="DFA")
plot(density(my.data$spread1),main="spread1")
plot(density(my.data$spread2),main="spread2")
plot(density(my.data$PPE),main="PPE")
par(op)
#' It appears that most of the predictor variables have distributions that lie close to the ideal shape of a normal distribution with the exception of "MDVP.Jitter...," "NHR," and "MDVP.Shimmer," which are right skewed. These variables could be transformed to more closely resemble normal distributions but, for the purpose of this analysis, the deviation from ideal will not be assumed significant enough to warrant correction. 
#' It is also useful to plot the distributions of the predictor variables by the binary response variable "status" in order to see how the distributions of values differ for each of the output classes. Use "plot_boxplot" from the DataExplorer package to achieve these kinds of distributions. Note for the "plot_boxplot" method that the response variable should be converted to a factor with two levels (1 and 0) for better visualization.
my.data$status = factor(my.data$status)
plot_boxplot(my.data,"status")
#' For many variables like spread1 and PPE, it appears that the distributions of values (median, interquartile range, etc.) are different for class "status"==0 (indicating patient is healthy) and class "status"==1 (indicating patient has Parkinson's), i.e. different ranges of values of the predictors correspond to the different classes. Therefore, using these variables as predictors would facilitate making classification decisions. 

#' createDataPartition() will now be used to split the dataset into a training set (used to train the model) and test set (used to evaluate the model's ability to generalize on unseen data), such that 75% of the dataset lies in the training set. For subsequent analysis, the response variable "status" must be numeric. 
my.data$status = as.numeric(as.character(my.data$status))
indices = createDataPartition(y=my.data$status,p=0.75,list=FALSE)
training.set = my.data[indices,]
test.set = my.data[-indices,]
nrow(training.set)
nrow(test.set)

#' The first model I will investigate is logistic regression, as covered in the Essential R notes and created solely using functions from base R. I will create a binary classification model using only one predictor (spread1) in order to examine how useful a simpler model would be (and see if a more complex model with multiple predictors is necessary). 
#' First I will plot "status" as a function of "spread1," using the jitter function to help visualize the points. 
with(training.set, plot(spread1,jitter(status,amount=0.04),ylab="",yaxt="n"))
axis(side=2, at=0:1, labels=c("0","1"))
#' Fit a logistic regression model using glm() on the training set
lr1 = glm(status~spread1, family="binomial", data=training.set)
summary(lr1)
#' Plot the logistic regression model to examine the fit
with(training.set, plot(spread1,jitter(status,amount=0.04),ylab="",yaxt="n"))
axis(side=2, at=0:1, labels=c("0","1"))
lines(cbind(training.set$spread1,lr1$fitted)[order(training.set$spread1),], col="red")
#' Plot the residual plots
op = par(mfrow=c(2,2),mar=c(4,4,2.5,0.5))
plot(lr1)
par(op)
#' There appears to be a pattern in the residuals vs fits plot, since "status" can only take a value of 0 or 1, and the model cannot predict intermediate values. 
#' Evaluate the model's overall performance by computing the p-value of a chi-squared statistic comparing the null and residual deviance. 
with(lr1, pchisq(null.deviance - deviance, df.null - df.residual, lower.tail=FALSE))
#' The p-value of is close to 0, indicating the model explains a significant amount of deviance. 
#' Use the roc() function from the pROC package to plot a ROC curve, a useful plot for binary classification problems that typically plots the true positive rate vs the false positive rate at different threshold settings, where the "threshold" is a variable boundary between the classes of interest. The "AUC" score, computed by auc() is a single number performance metric used to summarize the results displayed in a ROC curve (with higher values indicating better performance).
roc.curve = roc(training.set$status,lr1$fitted)
plot(roc.curve)
auc(roc.curve)
#' The AUC score of 0.8957 is high, but the ideal value is 1.0 and greater performance can be achieved by using multiple predictors (instead of solely the one predictor "spread1").

#' I will now use the caret package to analyze a series of different binary classification models (all now containing multiple predictors), with the goal of determining the most optimal one for this particular problem. The "caret" package is chosen for its streamlined interface, which enables quick prototyping and tuning of parameters for different models. For the subsequent analysis, the "status" variable must be converted to a factor and the names of the levels changed to "Zero" and "One" instead of "0" and "1," respectively, in order to conform to standards for variable naming in R. 
training.set$status = factor(training.set$status)
levels(training.set$status) = c("Zero","One")
test.set$status = factor(test.set$status)
levels(test.set$status) = c("Zero","One")
#' The primary function in caret is train() which trains a specified (binary classification) model on the dataset. A helper function called trainControl() is used to modify how train() operates on the dataset, i.e. it determines what hyperparameters are used to set up the model and how those hyperparameters are evaluated to determine the most optimal combination. (Hyperparameters are parameters not calculated during training, but must instead be pre-set by the user and optimized through experimentation. In support vector machines, for example, the cost parameter C is a hyperparameter used to determine how large coefficients in the model can be. Each classification model has a different set of hyperparameters that require tuning). 
trControl = trainControl(method = "repeatedcv",    
                         number = 5,
                         repeats = 3,
                         classProbs = TRUE,        
                         summaryFunction = defaultSummary)
#' In the arguments to trainControl(), I specified that repeated k-fold cross-validation (with 5 folds and 3 repeats) will be used to evaluate the optimal set of hyperparameters. Repeated cross validation involves splitting the training set into k folds or partitions, and the following procedure conducted for each fold: after creating the model with the desired combination of hyperparameters, (k-1) folds are used for training and the remaining unseen fold is used for testing or "validation". A "defaultSummary" will be used to display the performance metrics of the model and the argument "classProbs" was set to TRUE so that class probabilities (instead of final class decisions) are output by the model. (Class probabilities are necessary for the construction of ROC curves and computation of AUC scores). Lastly, for the purpose of simplicity, the default tuning grid of hyperparameters used by trainControl() and train() will be used. In the final analysis, a custom tuning grid will be created to further fine-tune the hyperparameter selection.
#' The train() function will now be used to fit a logistic regression model on all predictor variables in the dataset (instead of only the predictor "spread1" as described previously) using the metric "Accuracy" to select the best set of hyperparameters. 
model = train(form=status~.,data=training.set, method="glm", family="binomial", metric="Accuracy", trControl=trControl)
summary(model)
#' predict() will be used to make predictions from the model on the test set.
pred = predict(object=model,newdata=test.set)
#' table() can be used to display how many predictions of "Zero" vs "One" were made by the model.
table(pred)
#' The confusionMatrix() function can be used to output important information regarding the performance of the model on the test set, such as the sensitivity, specificity, and accuracy. The function also displays a confusion matrix of the true positive rate, true negative rate, false positive rate, and false negative rate. 
conf.matrix = confusionMatrix(data=pred,reference=test.set[,'status'])
conf.matrix
pred = predict(object=model,newdata=test.set,type="prob")
roc.curve = roc(test.set$status=="One",pred$One)
plot(roc.curve)
auc(roc.curve)

#' A Naive Bayes model will now be fit on the dataset. "Naive Bayes" is a probabilistic classification model that relies upon the assumption that all features are independent. 
model = train(form=status~.,data=training.set, method="naive_bayes", metric="Accuracy", trControl = trControl)
summary(model)
pred = predict(object=model,newdata=test.set)
table(pred)
conf.matrix = confusionMatrix(data=pred,reference=test.set[,'status'])
conf.matrix
#' This model achieves higher sensitivity, lower specificity, and lower accuracy compared to the logistic regression model and will therefore not be considered further.
pred = predict(object=model,newdata=test.set,type="prob")
roc.curve = roc(test.set$status=="One",pred$One)
plot(roc.curve)
auc(roc.curve)

#' A penalized logistic regression model will now be investigated. This model adds a penalty term to constrain the magnitude of the coefficients, thereby reducing the model's variance (letting the coefficient estimates be more stable).
model = train(form=status~.,data=training.set, method="plr", metric="Accuracy", trControl = trControl)
summary(model)
pred = predict(object=model,newdata=test.set)
table(pred)
conf.matrix = confusionMatrix(data=pred,reference=test.set[,'status'])
conf.matrix
#' This model achieves the same sensitivity, specificity, and accuracy as the original logistic regression model, and will not be a candidate for further investigation.
pred = predict(object=model,newdata=test.set,type="prob")
roc.curve = roc(test.set$status=="One",pred$One)
plot(roc.curve)
auc(roc.curve)

#' A boosted logistic regression model will now be investigated. This model uses a technique called "boosting," whereby an ensemble of weak learners is trained sequentially, thereby decreasing bias and more heavily weighting misclassified examples from previous base learners.Instead of the classical exponential loss used in the "Adaboost" algorithm, boosted logistic regression implements the log-loss as used in logistic regression.
model = train(form=status~.,data=training.set, method="LogitBoost", metric="Accuracy", trControl = trControl)
summary(model)
pred = predict(object=model,newdata=test.set)
table(pred)
conf.matrix = confusionMatrix(data=pred,reference=test.set[,'status'])
conf.matrix
#' This model achieves a higher sensitivity and accuracy than the original logistic regression model, and will therefore not be considered a candidate for further investigation. 
pred = predict(object=model,newdata=test.set,type="prob")
roc.curve = roc(test.set$status=="One",pred$One)
plot(roc.curve)
auc(roc.curve)

#' The last model of interest is a random forest model, constructed by creating an ensemble of decision trees where the final prediction is a weighted average or majority vote of the ensemble's predictions, thereby reducing the variance problem faced by single decision trees. 
model = train(form=status~.,data=training.set, method="ranger", metric="Accuracy", trControl = trControl)
summary(model)
pred = predict(object=model,newdata=test.set)
table(pred)
conf.matrix = confusionMatrix(data=pred,reference=test.set[,'status'])
conf.matrix
#' This model achieved a higher accuracy and specificity compared to the previous "best" model (the boosted logistic regression model). The random forest model will therefore be investigated further with a custom tuning grid.
pred = predict(object=model,newdata=test.set,type="prob")
roc.curve = roc(test.set$status=="One",pred$One)
plot(roc.curve)
auc(roc.curve)

#' There are three tuning parameters for the random forest: "mtry" (the number of features to look at when determining the split at each node of a decision tree in the ensemble), "splitrule" (the metric used at each node to determine the optimal split), and "min.node.size" (the minimum number of points from the dataset associated with each node, such that larger values of min.node.size correspond to smaller trees). A custom tuning grid will be created to fine tune the values of these three hyperparameters. Values for "mtry" from 2 through 20 and values of min.node.size taken from the set (2,5,10) will be investigated. The splitrule "extratrees" will be used in all hyperparameter combinations for simplicity, which determines the split point randomly instead of optimally based on a metric like "Gini."
tuneGrid = expand.grid(mtry = c(2:22),
                       splitrule = c("extratrees"),
                       min.node.size = c(1, 5, 10))
tuneGrid
#' Use train() with the new tuneGrid to determine the best combination of hyperparameters using the metric "Accuracy."
train.by.grid = train(form=status~.,data=training.set,method='ranger',metric = "Accuracy", tuneGrid=tuneGrid,trControl=trControl)
train.by.grid['results']
#' Print the optimal combination of hyperparameters
train.by.grid['bestTune']
# plot() can be used on the fitted model train.by.grid to graphically display the value of accuracy corresponding to each hyperparameter combination. 
plot(train.by.grid)
#' It appears that a minimal node size of 1 generally has the highest value of accuracy for values of mtry between 7 and 15.
pred = predict(object=train.by.grid,newdata=test.set)
table(pred)
conf.matrix = confusionMatrix(data=pred,reference=test.set[,'status'])
conf.matrix
pred = predict(object=train.by.grid,newdata=test.set,type="prob")
roc.curve = roc(test.set$status=="One",pred$One)
plot(roc.curve)

#' Fine-tune the values of the hyperparameters by focusing on their identified optimal ranges. 
tuneGrid = expand.grid(mtry = seq(7,15,0.5),
                       splitrule = c("extratrees"),
                       min.node.size = c(1,2,5))

tuneGrid
train.by.grid = train(form=status~.,data=training.set,method='ranger',metric="Accuracy",tuneGrid=tuneGrid,trControl=trControl)
train.by.grid['results']
#' Print the best hyperparameter combination
train.by.grid['bestTune']
#' Graphically compare the best hyperparameter combination with the other investigated possibilities. 
plot(train.by.grid)
pred = predict(object=train.by.grid,newdata=test.set)
table(pred)
conf.matrix = confusionMatrix(data=pred,reference=test.set[,'status'])
conf.matrix
pred = predict(object=train.by.grid,newdata=test.set,type="prob")
roc.curve = roc(test.set$status=="One",pred$One)
plot(roc.curve)
auc(roc.curve)
#' The best binary classification model is therefore random forest with a combination of "mtry," "splitrule," and "min.node.size" that can be tuned even further for optimal performance.
