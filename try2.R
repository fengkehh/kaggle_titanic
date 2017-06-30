library('lattice')
library('stringr')
library('mice')
library('caret')
library('parallel')
library('doParallel')

data <- read.csv('train.csv', header = TRUE, na.string = '')
n <- nrow(data)

# Data processing function
feature_engineer <- function(data) {
    
    # Need to change all factor levels to text for ensemble (classProbs = TRUE)
    
    # Process PClass levels
    class <- as.character(data$Pclass)
    
    class[class == '1'] <- 'First'
    class[class == '2'] <- 'Second'
    class[class == '3'] <- 'Third'
    
    data$Pclass <- class
    
    # Process title
    grep_str <- '[A-za-z]*\\.'
    
    titles <- str_match(data$Name, grep_str)
    
    titles[titles == 'Ms.'] <- 'Miss.'
    
    titles[titles == 'Mme.'] <- 'Mrs.'
    
    titles[titles == 'Mlle.'] <- 'Miss.'
    
    titles[titles == 'Dona.'] <- 'Mrs.'
        
    data$Title <- titles
    
    ## Extract decks.
    
    Mode <- function(x) {
        ux <- unique(x)
        ux[which.max(tabulate(match(x, ux)))]
    }
    
    tempfunc <- function(Cabin) {
        # Extract deck (if available)
        decks <- strsplit(Cabin, split = ' ')
        
        ans <- 'nodeck'
        
        if (!is.na(decks[[1]])) {
            deck_vec <- rep('', length(decks))
            for (ind in 1:length(decks)) {
                deck_vec[ind] = substr(decks[[ind]], 1, 1)
            }
            ans <- Mode(deck_vec)
        }
        
        return(ans)
    }
    
    deck_vec <- sapply(as.character(data$Cabin), FUN = tempfunc)
    data$Deck <- deck_vec
    
    # Process Survived levels
    if ('Survived' %in% names(data)) {
        survive <- as.character(data$Survived)
        
        survive[survive == '0'] <- 'dead'
        survive[survive == '1'] <- 'alive'
        
        data$Survived <- survive
        
        # Categorical variables
        categorical <- c('Sex', 'Survived', 'Pclass', 'Embarked', 'Name', 'Cabin', 
                         'PassengerId', 'Ticket', 'Title', 'Deck')
    } else {
        # Categorical variables
        categorical <- c('Sex', 'Pclass', 'Embarked', 'Name', 'Cabin', 
                         'PassengerId', 'Ticket', 'Title', 'Deck')
    }
    
    
    for (var in categorical) {
        data[, var] <- factor(data[, var])
    }
    
    return(data)
}

data.processed <- feature_engineer(data)

# 80%, 20% training/validation split
set.seed(123)
inTrain <- sample(1:n, size = ceiling(n*0.8))

data.train <- data.processed[inTrain, ]
data.validation <- data.processed[-inTrain, ]


# Uselss features: Name, Cabin, PassengerId
useless <- c('Name', 'Cabin', 'PassengerId', 'Ticket')

# Categorical variables: Sex, Survived, Pclass, Embarked
categorical <- c('Sex', 'Survived', 'Pclass', 'Embarked', 'Name', 'Cabin', 'PassengerId', 'Ticket', 'Title', 'Deck')

# All potentially useful predictors
useful <- names(data.train)[!(names(data.train) %in% 
                                  c(useless, 'Survived'))]

# Imputation
my_impute <- function(data) {
    data.useful <- data[,names(data) %in% c(useful, 'Survived')]
    #data.predictors <- data.useful[, !(names(data.useful) %in% c('Survived'))]
    
    missing <- is.na(data.useful$Embarked)
    embark_vec <- as.character(data.useful$Embarked)
    embark_vec[missing] <- 'missing'
    data.useful$Embark <- factor(embark_vec)
    
    mids <- mice(data.useful, m = 1, maxit = 20)
    
    data.useful.imp <- complete(mids)
    # if ('Survived' %in% names(data)) {
    #     data.imp <- data.frame(data.useful.imp, Survived = data$Survived)
    # } else {
    #     data.imp <- data.useful.imp
    # }
    
    return(data.useful.imp)
}

impute_mids <- function(data) {
    data.useful <- data[,names(data) %in% c(useful, 'Survived')]
    #data.predictors <- data.useful[, !(names(data.useful) %in% c('Survived'))]
    
    missing <- is.na(data.useful$Embarked)
    embark_vec <- as.character(data.useful$Embarked)
    embark_vec[missing] <- 'missing'
    data.useful$Embarked <- factor(embark_vec)
    
    mids <- mice(data.useful, m = 15, maxit = 20)
    
    return(mids)
}

my_preProc <- function(preProc = NULL, data) {
    if (is.null(preProc)) {
        # Preprocess
        preProc <- preProcess(data, method = c('center', 'scale', 'YeoJohnson'))
        
        return(preProc)
    } else {
        data.preproc <- predict(preProc, newdata = data)
        
        return(data.preproc)
    }
}

#data.train.imp <- my_impute(data.train)

# preProc <- my_preProc(data = data.train.imp)
# 
# data.train.imp.preproc <- my_preProc(preProc, data = data.train.imp)

# Build bagged trees on training set.
#model.treebag <- train(Survived ~. -Deck, data = data.train.imp, method = 'treebag', preProcess = c('center', 'scale', 'YeoJohnson'))

# Imputation
data.train.mids <- impute_mids(data.train)

data.train.imp1 <- complete(data.train.mids, 1)

# Generate preProc model for imputed data
preProc <- my_preProc(data = data.train.imp1)
data.train.imp1.preproc <- predict(preProc, data.train.imp1)


# Start training models
set.seed(123)
indices <- createResample(data.train.imp1$Survived, 25)
trCon <- trainControl(method = 'boot', number = 25, 
                      index = indices, 
                      savePredictions = TRUE,
                      classProbs = TRUE)

# Bagged trees model
set.seed(123)
model.treebag <- train(Survived ~., data = data.train.imp1.preproc, method = 'treebag', trControl = trCon)

# Logistic Regression
set.seed(123)
model.glmnet <- train(Survived ~., data = data.train.imp1.preproc, 
                      method = 'glmnet', 
                      trControl = trCon, 
                      family = 'binomial', 
                      standardize = FALSE)

# Build SVM model
set.seed(123)
model.svmPoly <- train(Survived ~., data = data.train.imp1.preproc, method = 'svmPoly', trControl = trCon, scaled = FALSE)

set.seed(123)
model.svmRadial <- train(Survived ~., data = data.train.imp1.preproc, method = 'svmRadial', trControl = trCon, scaled = FALSE)

set.seed(123)
model.svmLinear <- train(Survived ~., data = data.train.imp1.preproc, method = 'svmLinear', trControl = trCon, scaled = FALSE)

# KNN
set.seed(123)
model.knn <- train(Survived ~., data = data.train.imp1.preproc, method = 'kknn', trControl = trCon, scale = FALSE)

# Performance evaluation on validation set

# Impute first
data.validation.imp <- my_impute(data.validation)

# Preproc
data.validation.imp.preproc <- my_preProc(preProc, data.validation.imp)

# Bagged Tree
pred.treebag <- predict(model.treebag, newdata = data.validation.imp.preproc)
confusionMatrix(pred.treebag, data.validation$Survived)

# Logistic Regression
pred.glmnet <- predict(model.glmnet, newdata = data.validation.imp.preproc)
confusionMatrix(pred.glmnet, data.validation$Survived)

# POLY SVM
pred.svmPoly <- predict(model.svmPoly, newdata = data.validation.imp.preproc)
confusionMatrix(pred.svmPoly, data.validation$Survived)

# RBF SVM
pred.svmRadial <- predict(model.svmRadial, newdata = data.validation.imp.preproc)
confusionMatrix(pred.svmRadial, data.validation$Survived)

# Linear SVM
pred.svmLinear <- predict(model.svmLinear, newdata = data.validation.imp.preproc)
confusionMatrix(pred.svmLinear, data.validation$Survived)

# knn
pred.knn <- predict(model.knn, newdata = data.validation.imp.preproc)
confusionMatrix(pred.knn, data.validation$Survived)

my_list <- list(model.treebag, model.glmnet, model.knn, model.svmLinear, model.svmPoly, model.svmRadial)
names(my_list) <- c('treebag', 'glmnet', 'knn', 'linear svm', 'poly svm', 'radial svm')

modelCor(resamples(my_list))

# Build Ensemble Model
# glmnet_spec <- caretModelSpec(method = 'glmnet', family = 'binomial', standardize = FALSE)
# svmRadial_spec <- caretModelSpec(method = 'svmRadial', scaled = FALSE)
# 
# model_list <- caretList(Survived ~., data = data.train.imp1.preproc, 
#                         trControl = trCon,
#                         methodList = c('treebag', 'glmnet', 'svmRadial'),
#                         tuneList = list(glmnet_spec, svmRadial_spec))

# xyplot(resamples(model_list[1:3]))
# modelCor(resamples(model_list[1:3]))

final_list <-my_list[c(1,2,3,5)]
names(final_list) <- NULL
class(final_list) <- 'caretList'

model.ensemble <- caretEnsemble(final_list)
pred.ensemble <- predict(model.ensemble, data.validation.imp.preproc)
confusionMatrix(pred.ensemble, data.validation.imp.preproc$Survived)

