library(tidyverse)
library(caret)
library(mlbench)
library(glmnet)
library(glmnetUtils)

data(GermanCredit)
GermanCredit <- as.tibble(GermanCredit)

set.seed(121)
trainIndex <-
  createDataPartition(GermanCredit$Class,
                      p = 0.8,
                      list = FALSE,
                      times = 1)
germanTrain <- GermanCredit[trainIndex, ]
germanTest <- GermanCredit[-trainIndex, ]
scaler <- preProcess(germanTrain, method = c("center", "scale"))
germanTrain <- predict(scaler, germanTrain)
germanTest <- predict(scaler, germanTest)

lr<-glmnet(Class ~ .,
           data = germanTrain,
           family = "binomial",
           na.action = na.omit)

Predictions <- predict(lr,
                       germanTest,
                       type = "class",
                       na.action = na.pass,
                       s = 0.01)
confusionMatrix(Predictions,
               germanTest$Class)