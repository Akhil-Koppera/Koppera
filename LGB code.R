## Importing packages

# This R environment comes with all of CRAN and many other helpful packages preinstalled.
# You can see which packages are installed by checking out the kaggle/rstats docker image: 
# https://github.com/kaggle/docker-rstats

library(tidyverse) # metapackage with lots of helpful functions

## Running code

# In a notebook, you can run a single code cell by clicking in the cell and then hitting 
# the blue arrow to the left, or by clicking in the cell and pressing Shift+Enter. In a script, 
# you can run code by highlighting the code you want to run and then clicking the blue arrow
# at the bottom of this window.

## Reading in files

# You can access files from datasets you've added to this kernel in the "../input/" directory.
# You can see the files added to this kernel by running the code below. 

list.files(path = "../input/ieee-fraud-detection")

## Saving data

# If you save any files or images, these will be put in the "output" directory. You 
# can see the output directory by committing and running your kernel (using the 
# Commit & Run button) and then checking out the compiled version of your kernel.

train_transaction<-read.csv("../input/ieee-fraud-detection/train_transaction.csv")
train_identity<-read.csv('../input/ieee-fraud-detection/train_identity.csv')
test_identity<-read.csv('../input/ieee-fraud-detection/test_identity.csv')
test_transaction<-read.csv('../input/ieee-fraud-detection/test_transaction.csv')


library(data.table)
library(lubridate)
library(pryr)
library(caret)
library(xgboost)
library(readr)
library(ggthemes)
library(gridExtra)
library(repr)
library(fastDummies)
library(tictoc)
library(dplyr)
library(lightgbm)


train_data<- train_transaction %>% left_join(train_identity)
rm(train_identity,train_transaction); invisible(gc())
test_data <- test_transaction %>% left_join(test_identity)
rm(test_identity,test_transaction) ; invisible(gc())
drop_col <- c('V300','V309','V111','C3','V124','V106','V125','V315','V134','V102','V123','V316','V113',
              'V136','V305','V110','V299','V289','V286','V318','V103','V304','V116','V29','V284','V293',
              'V137','V295','V301','V104','V311','V115','V109','V119','V321','V114','V133','V122','V319',
              'V105','V112','V118','V117','V121','V108','V135','V320','V303','V297','V120')
train = train_data[,!(names(train_data) %in% drop_col)]
test = test_data[,!(names(test_data) %in% drop_col)]
rm(train_data,test_data) ; invisible(gc())
rm(drop_col);invisible(gc())


char_fea<-c("ProductCD","card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain",
            "R_emaildomain","M1","M2","M3","M4","M5","M6","M7","M8","M9","DeviceType","DeviceInfo","id_12",
            "id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24",
            "id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36",
            "id_37","id_38")

for(a in char_fea){
  train[,a]<-as.integer(train[,a])
  test[,a]<-as.integer(test[,a])
}
rm(char_fea);invisible(gc())

cat("train_col :" , ncol(train), "test_col :", ncol(test) ,"\n" )

y<-train$isFraud
train$isFraud <- NULL
tr_idx <- which(train$TransactionDT < quantile(train$TransactionDT,0.8))
#drop_cols1<-c('TransactionID', 'isFraud')
drop_cols1<-c('TransactionID', 'isFraud')
d0 <- lgb.Dataset(data.matrix( train[tr_idx,] ), label = y[tr_idx] )
dval <- lgb.Dataset(data.matrix( train[-tr_idx,!(names(train) %in% drop_cols1)] ), label = train[-tr_idx,'isFraud'] ) 
valids<-list(valid=dval)
lgb_param <- list(boosting_type = 'gbdt',
                  objective = "binary" ,
                  metric = "AUC",
                  boost_from_average = "false",
                  tree_learner  = "serial",
                  max_depth = -1,
                  learning_rate = 0.01,
                  num_leaves = 192,
                  min_gain_to_split = 0,
                  feature_fraction = 0.3,
                  feature_fraction_seed = 666666,
                  bagging_freq = 1,
                  bagging_fraction = 0.8,
                  min_sum_hessian_in_leaf = 0,
                  min_data_in_leaf = 100,
                  max_bin=155,
                  bagging_seed=11,
                  verbosity =-1
)


lgb <- lgb.train(params = lgb_param,  data = d0, nrounds = 15000, 
                 eval_freq = 200, valids = valids, early_stopping_rounds = 400, verbose = 1,seed=123)
library(tidyverse)
library(MLmetrics)
oof_pred <- predict(lgb, data.matrix(train[-tr_idx,]))
cat("best iter :" , lgb$best_iter, "best score :", AUC(oof_pred, y[-tr_idx]) ,"\n" )
iter <- lgb$best_iter
d0 <- lgb.Dataset( data.matrix( train ), label = y )
lgb <- lgb.train(params = lgb_param, data = d0, nrounds = iter*1.1, verbose = -1)
pred <- predict(lgb, data.matrix(test))

submission <- test[,'TransactionID' ]
submission<-cbind(submission,pred)
x<-data.frame(submission)
names(x)[names(x) == "submission"] <- "TransactionID"
names(x)[names(x) == "pred"] <- "isFraud"
submission<-x
rm(x)

submission$isFraud<-ifelse(submission$isFraud>0.5,1,0)
submission$TransactionID<-as.integer(submission$TransactionID)
#submission$TransactionID<-as.integer(submission$TransactionID)
write.csv(submission, file='submission.csv', row.names=FALSE)
lgb.save(lgb, "lgb.model")
imp <- lgb.importance(lgb)






