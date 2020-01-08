########################################################################################
################# Statistical Data Mining - Final Project###############################
########### Kaggel data set: IEEE Fraud Detection  using XGBOOST  ######################
########################################################################################



#rm(list = ls()) 

# set the current working directory
setwd("F:/Statistical Data Mining/Project")

#######################################
####### Installing packages############
#######################################
#install.packages('tidyverse')
#install.packages('ggthemes')
#install.packages('ggmosaic')
#install.packages('gridExtra')
#install.packages('repr')
#install.packages('data.table')
#install.packages('fastDummies')
#install.packages('xgboost')
#install.packages('tictoc')
#install.packages('readr')
#install.packages('tidyverse')
#install.packages('data.table')
#install.packages('lubridate')
#install.packages('pryr')
#install.packages('caret')
#install.packages('xgboost')



#######################################
###### Loading Libraries ##############
#######################################
library(tidyverse)
library(data.table)
library(lubridate)
library(pryr)
library(caret)
library(xgboost)
library(readr)
library(tidyverse)
library(ggthemes)
library(ggmosaic)
library(gridExtra)
library(repr)
library(data.table)
library(fastDummies)
library(xgboost)
library(tictoc)
library(dplyr)

#########################################################
######### Loadind the csv files #########################
#########################################################

train_transaction<-read.csv(file='train_transaction.csv',header=TRUE)
train_identity<-read.csv(file='train_identity.csv',header=TRUE)
test_identity<-read.csv(file='test_identity.csv', header=TRUE)
test_transaction<-read.csv(file='test_transaction.csv', header=TRUE)

##############################################################
####### creating test and train Data sets ####################
##############################################################
y <- train_transaction$isFraud
train$isFraud <- y
train_data<- train_transaction %>% left_join(train_identity)
target_df <- table(train_data[,'isFraud'])
barplot(target_df,main="Frequancy of Fraud",xlab="isFraud",ylab="Frequancy")
rm(train_identity,train_transaction); invisible(gc())
test_data <- test_transaction %>% left_join(test_identity)
rm(test_identity,test_transaction) ; invisible(gc())

#############################################################
##### Droping insignificant factors #########################
#############################################################


drop_col <- c('V300','V309','V111','C3','V124','V106','V125','V315','V134','V102','V123','V316','V113',
              'V136','V305','V110','V299','V289','V286','V318','V103','V304','V116','V29','V284','V293',
              'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
              'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120')

dim(train_data)
dim(test_data)
train = train_data[,!(names(train_data) %in% drop_col)]
test = test_data[,!(names(test_data) %in% drop_col)]
rm(train_data,test_data) ; invisible(gc())
rm(drop_col);invisible(gc())

############################################################
###### Converting the Cherecters to Numeric values #########
############################################################

char_fea<-c("ProductCD","card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain",
            "R_emaildomain","M1","M2","M3","M4","M5","M6","M7","M8","M9","DeviceType","DeviceInfo","id_12",
            "id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24",
            "id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36",
            "id_37","id_38")

for(a in char_fea){
  train[,a]<-as.numeric(train[,a])
  test[,a]<-as.numeric(test[,a])
}
rm(char_fea);invisible(gc())


#############################################################
######### Creating a xgboost model ##########################
#############################################################

drop_cols1<-c('TransactionID', 'isFraud')
dtrain <- xgb.DMatrix(data=as.matrix(train[,!(names(train) %in% drop_cols1)]), label=train[, 'isFraud'])
bst <- xgboost(data=dtrain,
               max_depth=10,
               eta=0.015,
               subsample=0.8,
               colsample_bytree=0.6,
               nthread=2,
               nrounds=1500,
               eval_metric='auc',
               objective='binary:logistic',
               verbose=1,
               verbose_eval=10)
xgb.save(bst, 'xgb.model')
rm(dtrain,train);invisible(gc())


###############################################################
####### Predictions based on the model created ################
###############################################################

pred <- predict(bst, as.matrix(test[,!(names(test) %in% drop_cols1)]))

rm(drop_cols1);invisible(gc())


################################################################
#### Creating a data frame and csv file of results #############
################################################################

submission <- test[,'TransactionID' ]
submission<-cbind(submission,pred)
x<-data.frame(submission)
names(x)[names(x) == "submission"] <- "TransactionID"
names(x)[names(x) == "pred"] <- "isFraud"
submission<-x
rm(x)

submission$isFraud<-ifelse(submission$isFraud>0.5,1,0)
submission$TransactionID<-as.integer(submission$TransactionID)


write.csv(submission, file='submission.csv', row.names=FALSE)



