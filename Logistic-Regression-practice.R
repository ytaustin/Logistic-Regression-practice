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

#round 1

lr_CreditHistory_Duration<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                    CreditHistory.ThisBank.AllPaid +
                                    CreditHistory.PaidDuly +
                                    CreditHistory.Delay +
                                    CreditHistory.Critical+
                                  Duration,
           data = germanTrain,
           family = "binomial",
           na.action = na.omit)

Predictions_CreditHistory_Duration <- predict(lr_CreditHistory_Duration,
                       germanTest,
                       type = "class",
                       na.action = na.pass,
                       s = 0.01)
confusionMatrix(Predictions_CreditHistory_Duration,
               germanTest$Class)

lr_CreditHistory_Amount<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                    CreditHistory.ThisBank.AllPaid +
                                    CreditHistory.PaidDuly +
                                    CreditHistory.Delay +
                                    CreditHistory.Critical+
                                    Amount,
                                  data = germanTrain,
                                  family = "binomial",
                                  na.action = na.omit)

Predictions_CreditHistory_Amount <- predict(lr_CreditHistory_Amount,
                                              germanTest,
                                              type = "class",
                                              na.action = na.pass,
                                              s = 0.01)
confusionMatrix(Predictions_CreditHistory_Amount,
                germanTest$Class)

lr_CreditHistory_InstallmentRatePercentage<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                  CreditHistory.ThisBank.AllPaid +
                                  CreditHistory.PaidDuly +
                                  CreditHistory.Delay +
                                  CreditHistory.Critical+
                                    InstallmentRatePercentage,
                                data = germanTrain,
                                family = "binomial",
                                na.action = na.omit)

Predictions_CreditHistory_InstallmentRatePercentage <- predict(lr_CreditHistory_InstallmentRatePercentage,
                                            germanTest,
                                            type = "class",
                                            na.action = na.pass,
                                            s = 0.01)
confusionMatrix(Predictions_CreditHistory_InstallmentRatePercentage,
                germanTest$Class)

lr_CreditHistory_Age<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                     CreditHistory.ThisBank.AllPaid +
                                                     CreditHistory.PaidDuly +
                                                     CreditHistory.Delay +
                                                     CreditHistory.Critical+
                                                     Age,
                                                   data = germanTrain,
                                                   family = "binomial",
                                                   na.action = na.omit)

Predictions_CreditHistory_Age <- predict(lr_CreditHistory_Age,
                                                               germanTest,
                                                               type = "class",
                                                               na.action = na.pass,
                                                               s = 0.01)
confusionMatrix(Predictions_CreditHistory_Age,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                               CreditHistory.ThisBank.AllPaid +
                               CreditHistory.PaidDuly +
                               CreditHistory.Delay +
                               CreditHistory.Critical+
                                 CheckingAccountStatus.lt.0 +
                                 CheckingAccountStatus.0.to.200 +
                                 CheckingAccountStatus.gt.200 +
                                 CheckingAccountStatus.none,
                             data = germanTrain,
                             family = "binomial",
                             na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus <- predict(lr_CreditHistory_CheckingAccountStatus,
                                         germanTest,
                                         type = "class",
                                         na.action = na.pass,
                                         s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus,
                germanTest$Class)

lr_CreditHistory_SavingsAccountBonds<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                 CreditHistory.ThisBank.AllPaid +
                                                 CreditHistory.PaidDuly +
                                                 CreditHistory.Delay +
                                                 CreditHistory.Critical+
                                               SavingsAccountBonds.lt.100 +
                                               SavingsAccountBonds.100.to.500 +
                                               SavingsAccountBonds.500.to.1000 +
                                               SavingsAccountBonds.gt.1000 +
                                               SavingsAccountBonds.Unknown,
                                               data = germanTrain,
                                               family = "binomial",
                                               na.action = na.omit)

Predictions_CreditHistory_SavingsAccountBonds <- predict(lr_CreditHistory_SavingsAccountBonds,
                                                           germanTest,
                                                           type = "class",
                                                           na.action = na.pass,
                                                           s = 0.01)
confusionMatrix(Predictions_CreditHistory_SavingsAccountBonds,
                germanTest$Class)

lr_CreditHistory_EmploymentDuration<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                               CreditHistory.ThisBank.AllPaid +
                                               CreditHistory.PaidDuly +
                                               CreditHistory.Delay +
                                               CreditHistory.Critical+
                                              EmploymentDuration.lt.1 +
                                              EmploymentDuration.1.to.4 +
                                              EmploymentDuration.4.to.7 +
                                              EmploymentDuration.gt.7 +
                                              EmploymentDuration.Unemployed,
                                             data = germanTrain,
                                             family = "binomial",
                                             na.action = na.omit)

Predictions_CreditHistory_EmploymentDuration <- predict(lr_CreditHistory_EmploymentDuration,
                                                         germanTest,
                                                         type = "class",
                                                         na.action = na.pass,
                                                         s = 0.01)
confusionMatrix(Predictions_CreditHistory_EmploymentDuration,
                germanTest$Class)


lr_CreditHistory_Personal<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                              CreditHistory.ThisBank.AllPaid +
                                              CreditHistory.PaidDuly +
                                              CreditHistory.Delay +
                                              CreditHistory.Critical+
                                    Personal.Male.Divorced.Seperated +
                                    Personal.Female.NotSingle +
                                    Personal.Male.Single +
                                    Personal.Male.Married.Widowed +
                                    Personal.Female.Single,
                                            data = germanTrain,
                                            family = "binomial",
                                            na.action = na.omit)

Predictions_CreditHistory_Personal <- predict(lr_CreditHistory_Personal,
                                                        germanTest,
                                                        type = "class",
                                                        na.action = na.pass,
                                                        s = 0.01)
confusionMatrix(Predictions_CreditHistory_Personal,
                germanTest$Class)

lr_CreditHistory_Housing<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                    CreditHistory.ThisBank.AllPaid +
                                    CreditHistory.PaidDuly +
                                    CreditHistory.Delay +
                                    CreditHistory.Critical+
                                   Housing.Rent +
                                   Housing.Own +
                                   Housing.ForFree,
                                  data = germanTrain,
                                  family = "binomial",
                                  na.action = na.omit)

Predictions_CreditHistory_Housing <- predict(lr_CreditHistory_Housing,
                                              germanTest,
                                              type = "class",
                                              na.action = na.pass,
                                              s = 0.01)
confusionMatrix(Predictions_CreditHistory_Housing,
                germanTest$Class)

#round 2

lr_CreditHistory_CheckingAccountStatus_Duration<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                          CreditHistory.ThisBank.AllPaid +
                                                          CreditHistory.PaidDuly +
                                                          CreditHistory.Delay +
                                                          CreditHistory.Critical+
                                                          CheckingAccountStatus.lt.0 +
                                                          CheckingAccountStatus.0.to.200 +
                                                          CheckingAccountStatus.gt.200 +
                                                          CheckingAccountStatus.none+
                                                          Duration,
                                  data = germanTrain,
                                  family = "binomial",
                                  na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Duration <- predict(lr_CreditHistory_CheckingAccountStatus_Duration,
                                              germanTest,
                                              type = "class",
                                              na.action = na.pass,
                                              s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Duration,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Amount<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                        CreditHistory.ThisBank.AllPaid +
                                                        CreditHistory.PaidDuly +
                                                        CreditHistory.Delay +
                                                        CreditHistory.Critical+
                                                        CheckingAccountStatus.lt.0 +
                                                        CheckingAccountStatus.0.to.200 +
                                                        CheckingAccountStatus.gt.200 +
                                                        CheckingAccountStatus.none+
                                                        Amount,
                                data = germanTrain,
                                family = "binomial",
                                na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Amount <- predict(lr_CreditHistory_CheckingAccountStatus_Amount,
                                            germanTest,
                                            type = "class",
                                            na.action = na.pass,
                                            s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Amount,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_InstallmentRatePercentage<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                           CreditHistory.ThisBank.AllPaid +
                                                                           CreditHistory.PaidDuly +
                                                                           CreditHistory.Delay +
                                                                           CreditHistory.Critical+
                                                                           CheckingAccountStatus.lt.0 +
                                                                           CheckingAccountStatus.0.to.200 +
                                                                           CheckingAccountStatus.gt.200 +
                                                                           CheckingAccountStatus.none+
                                                                           InstallmentRatePercentage,
                                                   data = germanTrain,
                                                   family = "binomial",
                                                   na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_InstallmentRatePercentage <- predict(lr_CreditHistory_CheckingAccountStatus_InstallmentRatePercentage,
                                                               germanTest,
                                                               type = "class",
                                                               na.action = na.pass,
                                                               s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_InstallmentRatePercentage,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Age<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                     CreditHistory.ThisBank.AllPaid +
                                                     CreditHistory.PaidDuly +
                                                     CreditHistory.Delay +
                                                     CreditHistory.Critical+
                                                     CheckingAccountStatus.lt.0 +
                                                     CheckingAccountStatus.0.to.200 +
                                                     CheckingAccountStatus.gt.200 +
                                                     CheckingAccountStatus.none+
                                                     Age ,
                             data = germanTrain,
                             family = "binomial",
                             na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Age <- predict(lr_CreditHistory_CheckingAccountStatus_Age,
                                         germanTest,
                                         type = "class",
                                         na.action = na.pass,
                                         s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Age,
                germanTest$Class)



lr_CreditHistory_CheckingAccountStatus_SavingsAccountBonds<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                     CreditHistory.ThisBank.AllPaid +
                                                                     CreditHistory.PaidDuly +
                                                                     CreditHistory.Delay +
                                                                     CreditHistory.Critical+
                                                                     CheckingAccountStatus.lt.0 +
                                                                     CheckingAccountStatus.0.to.200 +
                                                                     CheckingAccountStatus.gt.200 +
                                                                     CheckingAccountStatus.none+
                                                                     SavingsAccountBonds.lt.100 +
                                                                     SavingsAccountBonds.100.to.500 +
                                                                     SavingsAccountBonds.500.to.1000 +
                                                                     SavingsAccountBonds.gt.1000 +
                                                                     SavingsAccountBonds.Unknown,
                                             data = germanTrain,
                                             family = "binomial",
                                             na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_SavingsAccountBonds <- predict(lr_CreditHistory_CheckingAccountStatus_SavingsAccountBonds,
                                                         germanTest,
                                                         type = "class",
                                                         na.action = na.pass,
                                                         s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_SavingsAccountBonds,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_EmploymentDuration<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                    CreditHistory.ThisBank.AllPaid +
                                                                    CreditHistory.PaidDuly +
                                                                    CreditHistory.Delay +
                                                                    CreditHistory.Critical+
                                                                    CheckingAccountStatus.lt.0 +
                                                                    CheckingAccountStatus.0.to.200 +
                                                                    CheckingAccountStatus.gt.200 +
                                                                    CheckingAccountStatus.none+
                                                                    EmploymentDuration.lt.1 +
                                                                    EmploymentDuration.1.to.4 +
                                                                    EmploymentDuration.4.to.7 +
                                                                    EmploymentDuration.gt.7 +
                                                                    EmploymentDuration.Unemployed,
                                            data = germanTrain,
                                            family = "binomial",
                                            na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_EmploymentDuration <- predict(lr_CreditHistory_CheckingAccountStatus_EmploymentDuration,
                                                        germanTest,
                                                        type = "class",
                                                        na.action = na.pass,
                                                        s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_EmploymentDuration,
                germanTest$Class)


lr_CreditHistory_CheckingAccountStatus_Personal<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                          CreditHistory.ThisBank.AllPaid +
                                                          CreditHistory.PaidDuly +
                                                          CreditHistory.Delay +
                                                          CreditHistory.Critical+
                                                          CheckingAccountStatus.lt.0 +
                                                          CheckingAccountStatus.0.to.200 +
                                                          CheckingAccountStatus.gt.200 +
                                                          CheckingAccountStatus.none+
                                                          Personal.Male.Divorced.Seperated +
                                                          Personal.Female.NotSingle +
                                                          Personal.Male.Single +
                                                          Personal.Male.Married.Widowed +
                                                          Personal.Female.Single,
                                  data = germanTrain,
                                  family = "binomial",
                                  na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Personal <- predict(lr_CreditHistory_CheckingAccountStatus_Personal,
                                              germanTest,
                                              type = "class",
                                              na.action = na.pass,
                                              s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Personal,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                         CreditHistory.ThisBank.AllPaid +
                                                         CreditHistory.PaidDuly +
                                                         CreditHistory.Delay +
                                                         CreditHistory.Critical+
                                                         CheckingAccountStatus.lt.0 +
                                                         CheckingAccountStatus.0.to.200 +
                                                         CheckingAccountStatus.gt.200 +
                                                         CheckingAccountStatus.none+
                                                         Housing.Rent +
                                                         Housing.Own +
                                                         Housing.ForFree,
                                 data = germanTrain,
                                 family = "binomial",
                                 na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing <- predict(lr_CreditHistory_CheckingAccountStatus_Housing,
                                             germanTest,
                                             type = "class",
                                             na.action = na.pass,
                                             s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing,
                germanTest$Class)

#round 3

lr_CreditHistory_CheckingAccountStatus_Housing_Duration<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                         CreditHistory.ThisBank.AllPaid +
                                                         CreditHistory.PaidDuly +
                                                         CreditHistory.Delay +
                                                         CreditHistory.Critical+
                                                         CheckingAccountStatus.lt.0 +
                                                         CheckingAccountStatus.0.to.200 +
                                                         CheckingAccountStatus.gt.200 +
                                                         CheckingAccountStatus.none+
                                                         Housing.Rent +
                                                         Housing.Own +
                                                         Housing.ForFree+
                                                        Duration,
                                                       data = germanTrain,
                                                       family = "binomial",
                                                       na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_Duration <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_Duration,
                                                                   germanTest,
                                                                   type = "class",
                                                                   na.action = na.pass,
                                                                   s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_Duration,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_Amount<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                  CreditHistory.ThisBank.AllPaid +
                                                                  CreditHistory.PaidDuly +
                                                                  CreditHistory.Delay +
                                                                  CreditHistory.Critical+
                                                                  CheckingAccountStatus.lt.0 +
                                                                  CheckingAccountStatus.0.to.200 +
                                                                  CheckingAccountStatus.gt.200 +
                                                                  CheckingAccountStatus.none+
                                                                  Housing.Rent +
                                                                  Housing.Own +
                                                                  Housing.ForFree+
                                                                Amount,
                                                                data = germanTrain,
                                                                family = "binomial",
                                                                na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_Amount <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_Amount,
                                                                            germanTest,
                                                                            type = "class",
                                                                            na.action = na.pass,
                                                                            s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_Amount,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                CreditHistory.ThisBank.AllPaid +
                                                                CreditHistory.PaidDuly +
                                                                CreditHistory.Delay +
                                                                CreditHistory.Critical+
                                                                CheckingAccountStatus.lt.0 +
                                                                CheckingAccountStatus.0.to.200 +
                                                                CheckingAccountStatus.gt.200 +
                                                                CheckingAccountStatus.none+
                                                                Housing.Rent +
                                                                Housing.Own +
                                                                Housing.ForFree+
                                                                  InstallmentRatePercentage,
                                                              data = germanTrain,
                                                              family = "binomial",
                                                              na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage,
                                                                          germanTest,
                                                                          type = "class",
                                                                          na.action = na.pass,
                                                                          s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_Age<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                   CreditHistory.ThisBank.AllPaid +
                                                                                   CreditHistory.PaidDuly +
                                                                                   CreditHistory.Delay +
                                                                                   CreditHistory.Critical+
                                                                                   CheckingAccountStatus.lt.0 +
                                                                                   CheckingAccountStatus.0.to.200 +
                                                                                   CheckingAccountStatus.gt.200 +
                                                                                   CheckingAccountStatus.none+
                                                                                   Housing.Rent +
                                                                                   Housing.Own +
                                                                                   Housing.ForFree+
                                                                                   Age,
                                                                                 data = germanTrain,
                                                                                 family = "binomial",
                                                                                 na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_Age <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_Age,
                                                                                             germanTest,
                                                                                             type = "class",
                                                                                             na.action = na.pass,
                                                                                             s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_Age,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_SavingsAccountBonds<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                             CreditHistory.ThisBank.AllPaid +
                                                             CreditHistory.PaidDuly +
                                                             CreditHistory.Delay +
                                                             CreditHistory.Critical+
                                                             CheckingAccountStatus.lt.0 +
                                                             CheckingAccountStatus.0.to.200 +
                                                             CheckingAccountStatus.gt.200 +
                                                             CheckingAccountStatus.none+
                                                             Housing.Rent +
                                                             Housing.Own +
                                                             Housing.ForFree+
                                                               SavingsAccountBonds.lt.100 +
                                                               SavingsAccountBonds.100.to.500 +
                                                               SavingsAccountBonds.500.to.1000 +
                                                               SavingsAccountBonds.gt.1000 +
                                                               SavingsAccountBonds.Unknown,
                                                           data = germanTrain,
                                                           family = "binomial",
                                                           na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_SavingsAccountBonds <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_SavingsAccountBonds,
                                                                       germanTest,
                                                                       type = "class",
                                                                       na.action = na.pass,
                                                                       s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_SavingsAccountBonds,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_EmploymentDuration<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                             CreditHistory.ThisBank.AllPaid +
                                                                             CreditHistory.PaidDuly +
                                                                             CreditHistory.Delay +
                                                                             CreditHistory.Critical+
                                                                             CheckingAccountStatus.lt.0 +
                                                                             CheckingAccountStatus.0.to.200 +
                                                                             CheckingAccountStatus.gt.200 +
                                                                             CheckingAccountStatus.none+
                                                                             Housing.Rent +
                                                                             Housing.Own +
                                                                             Housing.ForFree+
                                                                            EmploymentDuration.lt.1 +
                                                                            EmploymentDuration.1.to.4 +
                                                                            EmploymentDuration.4.to.7 +
                                                                            EmploymentDuration.gt.7 +
                                                                            EmploymentDuration.Unemployed,
                                                                           data = germanTrain,
                                                                           family = "binomial",
                                                                           na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_EmploymentDuration <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_EmploymentDuration,
                                                                                       germanTest,
                                                                                       type = "class",
                                                                                       na.action = na.pass,
                                                                                       s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_EmploymentDuration,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_Personal<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                            CreditHistory.ThisBank.AllPaid +
                                                                            CreditHistory.PaidDuly +
                                                                            CreditHistory.Delay +
                                                                            CreditHistory.Critical+
                                                                            CheckingAccountStatus.lt.0 +
                                                                            CheckingAccountStatus.0.to.200 +
                                                                            CheckingAccountStatus.gt.200 +
                                                                            CheckingAccountStatus.none+
                                                                            Housing.Rent +
                                                                            Housing.Own +
                                                                            Housing.ForFree+
                                                                  Personal.Male.Divorced.Seperated +
                                                                  Personal.Female.NotSingle +
                                                                  Personal.Male.Single +
                                                                  Personal.Male.Married.Widowed +
                                                                  Personal.Female.Single,
                                                                          data = germanTrain,
                                                                          family = "binomial",
                                                                          na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_Personal <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_Personal,
                                                                                      germanTest,
                                                                                      type = "class",
                                                                                      na.action = na.pass,
                                                                                      s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_Personal,
                germanTest$Class)

#round 4

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Duration<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                   CreditHistory.ThisBank.AllPaid +
                                                                                   CreditHistory.PaidDuly +
                                                                                   CreditHistory.Delay +
                                                                                   CreditHistory.Critical+
                                                                                   CheckingAccountStatus.lt.0 +
                                                                                   CheckingAccountStatus.0.to.200 +
                                                                                   CheckingAccountStatus.gt.200 +
                                                                                   CheckingAccountStatus.none+
                                                                                   Housing.Rent +
                                                                                   Housing.Own +
                                                                                   Housing.ForFree+
                                                                                   InstallmentRatePercentage+
                                                                                     Duration,
                                                                                 data = germanTrain,
                                                                                 family = "binomial",
                                                                                 na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Duration <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Duration,
                                                                                             germanTest,
                                                                                             type = "class",
                                                                                             na.action = na.pass,
                                                                                             s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Duration,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                            CreditHistory.ThisBank.AllPaid +
                                                                                            CreditHistory.PaidDuly +
                                                                                            CreditHistory.Delay +
                                                                                            CreditHistory.Critical+
                                                                                            CheckingAccountStatus.lt.0 +
                                                                                            CheckingAccountStatus.0.to.200 +
                                                                                            CheckingAccountStatus.gt.200 +
                                                                                            CheckingAccountStatus.none+
                                                                                            Housing.Rent +
                                                                                            Housing.Own +
                                                                                            Housing.ForFree+
                                                                                            InstallmentRatePercentage+
                                                                                            Amount,
                                                                                          data = germanTrain,
                                                                                          family = "binomial",
                                                                                          na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount,
                                                                                                      germanTest,
                                                                                                      type = "class",
                                                                                                      na.action = na.pass,
                                                                                                      s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Age<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                          CreditHistory.ThisBank.AllPaid +
                                                                                          CreditHistory.PaidDuly +
                                                                                          CreditHistory.Delay +
                                                                                          CreditHistory.Critical+
                                                                                          CheckingAccountStatus.lt.0 +
                                                                                          CheckingAccountStatus.0.to.200 +
                                                                                          CheckingAccountStatus.gt.200 +
                                                                                          CheckingAccountStatus.none+
                                                                                          Housing.Rent +
                                                                                          Housing.Own +
                                                                                          Housing.ForFree+
                                                                                          InstallmentRatePercentage+
                                                                                          Age,
                                                                                        data = germanTrain,
                                                                                        family = "binomial",
                                                                                        na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Age <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Age,
                                                                                                    germanTest,
                                                                                                    type = "class",
                                                                                                    na.action = na.pass,
                                                                                                    s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Age,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_SavingsAccountBonds<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                       CreditHistory.ThisBank.AllPaid +
                                                                                       CreditHistory.PaidDuly +
                                                                                       CreditHistory.Delay +
                                                                                       CreditHistory.Critical+
                                                                                       CheckingAccountStatus.lt.0 +
                                                                                       CheckingAccountStatus.0.to.200 +
                                                                                       CheckingAccountStatus.gt.200 +
                                                                                       CheckingAccountStatus.none+
                                                                                       Housing.Rent +
                                                                                       Housing.Own +
                                                                                       Housing.ForFree+
                                                                                       InstallmentRatePercentage+
                                                                                         SavingsAccountBonds.lt.100 +
                                                                                         SavingsAccountBonds.100.to.500 +
                                                                                         SavingsAccountBonds.500.to.1000 +
                                                                                         SavingsAccountBonds.gt.1000 +
                                                                                         SavingsAccountBonds.Unknown,
                                                                                     data = germanTrain,
                                                                                     family = "binomial",
                                                                                     na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_SavingsAccountBonds <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_SavingsAccountBonds,
                                                                                                 germanTest,
                                                                                                 type = "class",
                                                                                                 na.action = na.pass,
                                                                                                 s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_SavingsAccountBonds,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_EmploymentDuration<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                                       CreditHistory.ThisBank.AllPaid +
                                                                                                       CreditHistory.PaidDuly +
                                                                                                       CreditHistory.Delay +
                                                                                                       CreditHistory.Critical+
                                                                                                       CheckingAccountStatus.lt.0 +
                                                                                                       CheckingAccountStatus.0.to.200 +
                                                                                                       CheckingAccountStatus.gt.200 +
                                                                                                       CheckingAccountStatus.none+
                                                                                                       Housing.Rent +
                                                                                                       Housing.Own +
                                                                                                       Housing.ForFree+
                                                                                                       InstallmentRatePercentage+
                                                                                                      EmploymentDuration.lt.1 +
                                                                                                      EmploymentDuration.1.to.4 +
                                                                                                      EmploymentDuration.4.to.7 +
                                                                                                      EmploymentDuration.gt.7 +
                                                                                                      EmploymentDuration.Unemployed,
                                                                                                     data = germanTrain,
                                                                                                     family = "binomial",
                                                                                                     na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_EmploymentDuration <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_EmploymentDuration,
                                                                                                                 germanTest,
                                                                                                                 type = "class",
                                                                                                                 na.action = na.pass,
                                                                                                                 s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_EmploymentDuration,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Personal<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                                      CreditHistory.ThisBank.AllPaid +
                                                                                                      CreditHistory.PaidDuly +
                                                                                                      CreditHistory.Delay +
                                                                                                      CreditHistory.Critical+
                                                                                                      CheckingAccountStatus.lt.0 +
                                                                                                      CheckingAccountStatus.0.to.200 +
                                                                                                      CheckingAccountStatus.gt.200 +
                                                                                                      CheckingAccountStatus.none+
                                                                                                      Housing.Rent +
                                                                                                      Housing.Own +
                                                                                                      Housing.ForFree+
                                                                                                      InstallmentRatePercentage+
                                                                                            Personal.Male.Divorced.Seperated +
                                                                                            Personal.Female.NotSingle +
                                                                                            Personal.Male.Single +
                                                                                            Personal.Male.Married.Widowed +
                                                                                            Personal.Female.Single,
                                                                                                    data = germanTrain,
                                                                                                    family = "binomial",
                                                                                                    na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Personal <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Personal,
                                                                                                                germanTest,
                                                                                                                type = "class",
                                                                                                                na.action = na.pass,
                                                                                                                s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Personal,
                germanTest$Class)

#round 5

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Duration<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                          CreditHistory.ThisBank.AllPaid +
                                                                                          CreditHistory.PaidDuly +
                                                                                          CreditHistory.Delay +
                                                                                          CreditHistory.Critical+
                                                                                          CheckingAccountStatus.lt.0 +
                                                                                          CheckingAccountStatus.0.to.200 +
                                                                                          CheckingAccountStatus.gt.200 +
                                                                                          CheckingAccountStatus.none+
                                                                                          Housing.Rent +
                                                                                          Housing.Own +
                                                                                          Housing.ForFree+
                                                                                          InstallmentRatePercentage+
                                                                                          Amount+
                                                                                            Duration,
                                                                                        data = germanTrain,
                                                                                        family = "binomial",
                                                                                        na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Duration <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Duration,
                                                                                                    germanTest,
                                                                                                    type = "class",
                                                                                                    na.action = na.pass,
                                                                                                    s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Duration,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                                   CreditHistory.ThisBank.AllPaid +
                                                                                                   CreditHistory.PaidDuly +
                                                                                                   CreditHistory.Delay +
                                                                                                   CreditHistory.Critical+
                                                                                                   CheckingAccountStatus.lt.0 +
                                                                                                   CheckingAccountStatus.0.to.200 +
                                                                                                   CheckingAccountStatus.gt.200 +
                                                                                                   CheckingAccountStatus.none+
                                                                                                   Housing.Rent +
                                                                                                   Housing.Own +
                                                                                                   Housing.ForFree+
                                                                                                   InstallmentRatePercentage+
                                                                                                   Amount+
                                                                                                   Age,
                                                                                                 data = germanTrain,
                                                                                                 family = "binomial",
                                                                                                 na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age,
                                                                                                             germanTest,
                                                                                                             type = "class",
                                                                                                             na.action = na.pass,
                                                                                                             s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_SavingsAccountBonds<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                              CreditHistory.ThisBank.AllPaid +
                                                                                              CreditHistory.PaidDuly +
                                                                                              CreditHistory.Delay +
                                                                                              CreditHistory.Critical+
                                                                                              CheckingAccountStatus.lt.0 +
                                                                                              CheckingAccountStatus.0.to.200 +
                                                                                              CheckingAccountStatus.gt.200 +
                                                                                              CheckingAccountStatus.none+
                                                                                              Housing.Rent +
                                                                                              Housing.Own +
                                                                                              Housing.ForFree+
                                                                                              InstallmentRatePercentage+
                                                                                              Amount+
                                                                                                SavingsAccountBonds.lt.100 +
                                                                                                SavingsAccountBonds.100.to.500 +
                                                                                                SavingsAccountBonds.500.to.1000 +
                                                                                                SavingsAccountBonds.gt.1000 +
                                                                                                SavingsAccountBonds.Unknown,
                                                                                            data = germanTrain,
                                                                                            family = "binomial",
                                                                                            na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_SavingsAccountBonds <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_SavingsAccountBonds,
                                                                                                        germanTest,
                                                                                                        type = "class",
                                                                                                        na.action = na.pass,
                                                                                                        s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_SavingsAccountBonds,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_EmploymentDuration<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                                              CreditHistory.ThisBank.AllPaid +
                                                                                                              CreditHistory.PaidDuly +
                                                                                                              CreditHistory.Delay +
                                                                                                              CreditHistory.Critical+
                                                                                                              CheckingAccountStatus.lt.0 +
                                                                                                              CheckingAccountStatus.0.to.200 +
                                                                                                              CheckingAccountStatus.gt.200 +
                                                                                                              CheckingAccountStatus.none+
                                                                                                              Housing.Rent +
                                                                                                              Housing.Own +
                                                                                                              Housing.ForFree+
                                                                                                              InstallmentRatePercentage+
                                                                                                              Amount+
                                                                                                             EmploymentDuration.lt.1 +
                                                                                                             EmploymentDuration.1.to.4 +
                                                                                                             EmploymentDuration.4.to.7 +
                                                                                                             EmploymentDuration.gt.7 +
                                                                                                             EmploymentDuration.Unemployed,
                                                                                                            data = germanTrain,
                                                                                                            family = "binomial",
                                                                                                            na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_EmploymentDuration <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_EmploymentDuration,
                                                                                                                        germanTest,
                                                                                                                        type = "class",
                                                                                                                        na.action = na.pass,
                                                                                                                        s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_EmploymentDuration,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Personal<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                                             CreditHistory.ThisBank.AllPaid +
                                                                                                             CreditHistory.PaidDuly +
                                                                                                             CreditHistory.Delay +
                                                                                                             CreditHistory.Critical+
                                                                                                             CheckingAccountStatus.lt.0 +
                                                                                                             CheckingAccountStatus.0.to.200 +
                                                                                                             CheckingAccountStatus.gt.200 +
                                                                                                             CheckingAccountStatus.none+
                                                                                                             Housing.Rent +
                                                                                                             Housing.Own +
                                                                                                             Housing.ForFree+
                                                                                                             InstallmentRatePercentage+
                                                                                                             Amount+
                                                                                                   Personal.Male.Divorced.Seperated +
                                                                                                   Personal.Female.NotSingle +
                                                                                                   Personal.Male.Single +
                                                                                                   Personal.Male.Married.Widowed +
                                                                                                   Personal.Female.Single,
                                                                                                           data = germanTrain,
                                                                                                           family = "binomial",
                                                                                                           na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Personal <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Personal,
                                                                                                                       germanTest,
                                                                                                                       type = "class",
                                                                                                                       na.action = na.pass,
                                                                                                                       s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Personal,
                germanTest$Class)

#round 6

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_Duration<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                              CreditHistory.ThisBank.AllPaid +
                                                                                              CreditHistory.PaidDuly +
                                                                                              CreditHistory.Delay +
                                                                                              CreditHistory.Critical+
                                                                                              CheckingAccountStatus.lt.0 +
                                                                                              CheckingAccountStatus.0.to.200 +
                                                                                              CheckingAccountStatus.gt.200 +
                                                                                              CheckingAccountStatus.none+
                                                                                              Housing.Rent +
                                                                                              Housing.Own +
                                                                                              Housing.ForFree+
                                                                                              InstallmentRatePercentage+
                                                                                              Amount+
                                                                                              Age+
                                                                                                Duration,
                                                                                            data = germanTrain,
                                                                                            family = "binomial",
                                                                                            na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_Duration <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_Duration,
                                                                                                        germanTest,
                                                                                                        type = "class",
                                                                                                        na.action = na.pass,
                                                                                                        s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_Duration,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_SavingsAccountBonds<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                                       CreditHistory.ThisBank.AllPaid +
                                                                                                       CreditHistory.PaidDuly +
                                                                                                       CreditHistory.Delay +
                                                                                                       CreditHistory.Critical+
                                                                                                       CheckingAccountStatus.lt.0 +
                                                                                                       CheckingAccountStatus.0.to.200 +
                                                                                                       CheckingAccountStatus.gt.200 +
                                                                                                       CheckingAccountStatus.none+
                                                                                                       Housing.Rent +
                                                                                                       Housing.Own +
                                                                                                       Housing.ForFree+
                                                                                                       InstallmentRatePercentage+
                                                                                                       Amount+
                                                                                                       Age+
                                                                                                         SavingsAccountBonds.lt.100 +
                                                                                                         SavingsAccountBonds.100.to.500 +
                                                                                                         SavingsAccountBonds.500.to.1000 +
                                                                                                         SavingsAccountBonds.gt.1000 +
                                                                                                         SavingsAccountBonds.Unknown,
                                                                                                     data = germanTrain,
                                                                                                     family = "binomial",
                                                                                                     na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_SavingsAccountBonds <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_SavingsAccountBonds,
                                                                                                                 germanTest,
                                                                                                                 type = "class",
                                                                                                                 na.action = na.pass,
                                                                                                                 s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_SavingsAccountBonds,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_EmploymentDuration<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                                                  CreditHistory.ThisBank.AllPaid +
                                                                                                                  CreditHistory.PaidDuly +
                                                                                                                  CreditHistory.Delay +
                                                                                                                  CreditHistory.Critical+
                                                                                                                  CheckingAccountStatus.lt.0 +
                                                                                                                  CheckingAccountStatus.0.to.200 +
                                                                                                                  CheckingAccountStatus.gt.200 +
                                                                                                                  CheckingAccountStatus.none+
                                                                                                                  Housing.Rent +
                                                                                                                  Housing.Own +
                                                                                                                  Housing.ForFree+
                                                                                                                  InstallmentRatePercentage+
                                                                                                                  Amount+
                                                                                                                  Age+
                                                                                                                 EmploymentDuration.lt.1 +
                                                                                                                 EmploymentDuration.1.to.4 +
                                                                                                                 EmploymentDuration.4.to.7 +
                                                                                                                 EmploymentDuration.gt.7 +
                                                                                                                 EmploymentDuration.Unemployed,
                                                                                                                data = germanTrain,
                                                                                                                family = "binomial",
                                                                                                                na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_EmploymentDurations <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_EmploymentDuration,
                                                                                                                            germanTest,
                                                                                                                            type = "class",
                                                                                                                            na.action = na.pass,
                                                                                                                            s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_EmploymentDurations,
                germanTest$Class)

lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_Personal<-glmnet(Class ~ CreditHistory.NoCredit.AllPaid +
                                                                                                                 CreditHistory.ThisBank.AllPaid +
                                                                                                                 CreditHistory.PaidDuly +
                                                                                                                 CreditHistory.Delay +
                                                                                                                 CreditHistory.Critical+
                                                                                                                 CheckingAccountStatus.lt.0 +
                                                                                                                 CheckingAccountStatus.0.to.200 +
                                                                                                                 CheckingAccountStatus.gt.200 +
                                                                                                                 CheckingAccountStatus.none+
                                                                                                                 Housing.Rent +
                                                                                                                 Housing.Own +
                                                                                                                 Housing.ForFree+
                                                                                                                 InstallmentRatePercentage+
                                                                                                                 Amount+
                                                                                                                 Age+
                                                                                                       Personal.Male.Divorced.Seperated +
                                                                                                       Personal.Female.NotSingle +
                                                                                                       Personal.Male.Single +
                                                                                                       Personal.Male.Married.Widowed +
                                                                                                       Personal.Female.Single,
                                                                                                               data = germanTrain,
                                                                                                               family = "binomial",
                                                                                                               na.action = na.omit)

Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_Personal <- predict(lr_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_Personal,
                                                                                                                            germanTest,
                                                                                                                            type = "class",
                                                                                                                            na.action = na.pass,
                                                                                                                            s = 0.01)
confusionMatrix(Predictions_CreditHistory_CheckingAccountStatus_Housing_InstallmentRatePercentage_Amount_Age_Personal,
                germanTest$Class)

#all features

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