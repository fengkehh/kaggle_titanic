library('lattice')
library('caret')
library('missForest')
library('parallel')
library('doParallel')

source('test_funcs.R')

data <- read.csv('train.csv', header = TRUE)

# Categorical variables: Sex, Survived, Pclass, Embarked
categorical <- c('Sex', 'Survived', 'Pclass', 'Embarked')

for (var in categorical) {
    data[,var] <- factor(data[,var])
}

# response = Survived

n <- nrow(data)

# Uselss features: Name, Cabin, PassengerId
useless <- c('Name', 'Cabin', 'PassengerId', 'Ticket')

# Current strategy at dealing with missing values: get rid of incomplete measurements.

# Extract all full data points
# data.full <- data[!missing_ind,]
# 
# n.full <- nrow(data.full)

# 80%, 20% training/testing split
set.seed(123)
# inTrain <- sample(1:n.full, size = ceiling(n.full*0.8))
# 
# data.train <- data.full[inTrain,]
# data.test <- data.full[-inTrain,]


inTrain <- sample(1:n, size = ceiling(n*0.8))

data.train <- data[inTrain, ]
data.test <- data[-inTrain, ]

# densityplot(~ Age, data = data.train)
# densityplot( ~ Parch, data = data.train)

# All potentially useful predictors
useful <- names(data.train)[!(names(data.train) %in% 
                                             c(useless, 'Survived'))]

# Useful numerical predictors
useful.numerical <- useful[!(useful %in% categorical)]

# check if any predictors should be removed due to colinearity
findCorrelation(x = cor(data.train[,useful.numerical]), names = TRUE, 
                cutoff = 0.9)


formula <- as.formula(paste('Survived', paste(useful, collapse = ' + '), 
                            sep = ' ~ '))

# Random Forest
num_trees <- 5001

train.preProc <- my_preProcess(data.train, useful)

tunegrid <- expand.grid(mtry = c(1:length(useful)))

trCon <- trainControl(method = 'cv', number = 10, search = 'grid',
                      allowParallel = TRUE)
# 
# cl <- makePSOCKcluster(detectCores())
# registerDoParallel(cl)
# 
# set.seed(123)
# rf_gridsearch <- train(formula, data = train.preProc$data, method = 'rf', 
#                        ntree = num_trees, tuneGrid = tunegrid, 
#                        importance = TRUE, na.action = na.fail, trControl = trCon) 
# 
# stopCluster(cl)
# 
# 
# pred <- my_predict(rf_gridsearch, data.test, useful)
# 
# confusionMatrix(pred, data.test$Survived)
# 
# varImpPlot(rf_gridsearch$finalModel)

## Do it with imputed data
set.seed(123)
imputation <- my_impute(data.train, useful)

print(imputation$obj.imp$OOBerror)

data.train.imp <- imputation$data

imputation <- my_impute(data.test, useful)

print(imputation$obj.imp$OOBerror)

data.test.imp <- imputation$data


# Train Random Forest model
cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)

set.seed(123)
trCon <- trainControl(method = 'cv', number = 10, search = 'grid', 
                      allowParallel = TRUE)
rf_gridsearch.imp <- train(formula, data = data.train.imp, method = 'rf', 
                       ntree = num_trees, tuneGrid = tunegrid, 
                       importance = TRUE, na.action = na.fail, trControl = trCon) 

stopCluster(cl)

pred.imp <- predict(rf_gridsearch.imp, data.test.imp)

confusionMatrix(pred.imp, data.test.imp$Survived)

varImpPlot(rf_gridsearch.imp$finalModel)
