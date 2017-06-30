data <- read.csv('train.csv', header = TRUE, na.string = '')

# Prepare full training set
full.processed <- feature_engineer(data)
full.imp <- my_impute(full.processed)
full.preProc <- my_preProc(data = full.imp)
full.imp.preproc <- predict(full.preProc, full.imp)

# Start training models
set.seed(123)
full.indices <- createResample(full.imp.preproc$Survived, 25)
full.trCon <- trainControl(method = 'boot', number = 25, 
                      index = full.indices, 
                      savePredictions = TRUE,
                      classProbs = TRUE)

# Bagged trees model
set.seed(123)
full.treebag <- train(Survived ~., data = full.imp.preproc, 
                      method = 'treebag', 
                      trControl = full.trCon)

# Logistic Regression
set.seed(123)
full.glmnet <- train(Survived ~., data = full.imp.preproc, 
                      method = 'glmnet', 
                      trControl = full.trCon, 
                      family = 'binomial', 
                      standardize = FALSE)

# Poly SVM
set.seed(123)
full.svmPoly <- train(Survived ~., data = full.imp.preproc, 
                       method = 'svmPoly', 
                       trControl = full.trCon, 
                       scaled = FALSE)

# KNN
set.seed(123)
full.knn <- train(Survived ~., data = full.imp.preproc, 
                   method = 'kknn', 
                   trControl = full.trCon, 
                   scale = FALSE)

full_list <- list(full.treebag, full.glmnet, full.knn, full.svmPoly)
class(full_list) <- 'caretList'

full.ensemble <- caretEnsemble(full_list)


# Prepare test set for prediction.
data.test <- read.csv('test.csv', header = TRUE, na.string = '')

test.processed <- feature_engineer(data.test)
test.imp <- my_impute(test.processed)
test.imp.preproc <- predict(full.preProc, test.imp)

# attach fake Survived to trick caretEnsemble into making predictions
# test.imp.preproc.hack <- data.frame(test.imp.preproc, Survived = rep(0, nrow(test.imp.preproc)))

test.pred <- as.character(predict(full.ensemble, test.imp.preproc))

test.pred[test.pred == 'dead'] <- 0
test.pred[test.pred == 'alive'] <- 1

df.out <- data.frame(PassengerId = data.test$PassengerId, Survived = test.pred)

write.csv(df.out, file = 'my_submission.csv', row.names = FALSE, quote = FALSE)

