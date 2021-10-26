#initialize packages
library(caret)
library(RTextTools)
library(stringi)
library(quanteda)
library(quanteda.textmodels)

#import master dataset of all movie reviews both training and submission that 
#have been combined into the "master_dataset.csv"
masterdata <- read.csv("master_dataset.csv")

#transform to dataframe object
master_df <- data.frame(masterdata)

#transform dataframe to corpus object and define vaiables 
corpus_master <- corpus(master_df, 
                        docid_field = "id",
                        text_field = "text",
                        meta = list(),
                        unique_docnames = TRUE)
#check number of columns
ncol(master_df)

set.seed(123)

#generate tokens from the corpus
toks_master <- tokens(corpus_master, remove_punct = TRUE)


dfm_svm <- dfm(toks_master)

#print(toks_master)
#print(dfm_svm)

#dim(toks_master)
#?dfm_subset

submission_dfm <- dfm_subset(dfm_svm, label == "NAN")
training_dfm <- dfm_subset(dfm_svm, (label == 1) | (label == 0))

#print(submission_dfm)
#docvars(submission_dfm)
#docvars(training_dfm)

##############
# NAIVE BAYES
# Submission to Kaggle at 13:45 GMT 30 March 2021
# results 0.66

nb_model <- textmodel_nb(training_dfm, y = docvars(training_dfm, "label"))

pred_nb <- predict(nb_model, newdata = submission_dfm)

#print(pred_nb)

predictions_string <- str(pred_nb)

write.table(predictions_string, 
            file = "predictions2.csv", 
            sep = ",",
            qmethod = "double",
            row.names = FALSE)

###########################
# SUPPORT VECTOR MACHINE
# submitted to Kaggle at 13:50 GMT on 30 March 2021
# results: 0.59

svm_model <- textmodel_svm(training_dfm, y = docvars(training_dfm, "label"))

pred_svm <- predict(svm_model, newdata = submission_dfm)

#print(pred_svm)

write.table(pred_svm, 
            file = "predictions3.csv", 
            sep = ",",
            qmethod = "double",
            row.names = FALSE)

################################
# SVM LINEAR
# Submitted to Kaggle at 13:58 GMT 30 March 2021
# Results 0.59  

svmlin_model <- textmodel_svmlin(training_dfm, y = docvars(training_dfm, "label"))
pred_svmlin <- predict(svm_model, newdata = submission_dfm)

#print(pred_svm)

write.table(pred_svmlin, 
            file = "predictions4.csv", 
            sep = ",",
            qmethod = "double",
            row.names = FALSE)

##############################################################################
# SUBMISSIONS WITH ALTERNATE FEATURE ENGINEERING & SAME ANALYSIS
# Submitted to Kaggle as "submissions5.csv" at 10:43 GMT on 31 March 2021
# Scored 

#convert tokens used above to lowercase, stem tokens, & remove stopwords
dfm2 <- dfm(toks_master,
            stem = TRUE,
            tolower = TRUE,
            remove = stopwords("english"))


submission_dfm2 <- dfm_subset(dfm2, label == "NAN")
training_dfm2 <- dfm_subset(dfm2, (label == 1) | (label == 0))


svm_model2 <- textmodel_svm(training_dfm2, y = docvars(training_dfm2, "label"))

pred_svm2 <- predict(svm_model2, newdata = submission_dfm2)

#print(pred_svm2)

write.table(pred_svm, 
            file = "predictions5.csv", 
            sep = ",",
            qmethod = "double",
            row.names = FALSE)

#######################################################
#NAIVE BAYES 
#Submitted to Kaggle as "predictions6.csv" at 10:48 GMT on 31 March 2021
# Score 0.6366 in Kaggle

nb_model2 <- textmodel_nb(training_dfm2, y = docvars(training_dfm2, "label"))

pred_nb2 <- predict(nb_model2, newdata = submission_dfm2)


write.table(pred_nb2, 
            file = "predictions6.csv", 
            sep = ",",
            qmethod = "double",
            row.names = FALSE)


###########################################################################
# Since Naive Bayes has performed the best, doing Naive Bayes anaylsis
# without tokenizing first, and doing so while creating the dfm and 
#removing stemming and removing punctuation
#Submitted to Kaggle at 11:26 GMT 31 March 2021
# Kaggle score 0.5666


dfm3 <- dfm(corpus_master,
            stem = FALSE,
            tolower = TRUE,
            remove = stopwords("english"),
            remove_punct = TRUE)

submission_dfm3 <- dfm_subset(dfm3, label == "NAN")
training_dfm3 <- dfm_subset(dfm3, (label == 1) | (label == 0))

nb_model3 <- textmodel_svm(training_dfm3, y = docvars(training_dfm3, "label"))

pred_nb3 <- predict(nb_model3, newdata = submission_dfm3)

#print(pred_nb3)

write.table(pred_nb3, 
            file = "predictions7.csv", 
            sep = ",",
            qmethod = "double",
            row.names = FALSE)
