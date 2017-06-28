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
    
    # Process title
    grep_str <- '[A-za-z]*\\.'
    
    titles <- str_match(data$Name, grep_str)
    
    titles[titles == 'Ms.'] <- 'Miss.'
    
    titles[titles == 'Mme.'] <- 'Mrs.'
    
    titles[titles == 'Mlle.'] <- 'Miss.'
    
    data$Title <- titles
    
    ## Extract decks.
    
    Mode <- function(x) {
        ux <- unique(x)
        ux[which.max(tabulate(match(x, ux)))]
    }
    
    tempfunc <- function(Cabin) {
        # Extract deck (if available)
        decks <- strsplit(Cabin, split = ' ')
        
        ans <- 'missing'
        
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
    
    # Categorical variables: Sex, Survived, Pclass, Embarked
    categorical <- c('Sex', 'Survived', 'Pclass', 'Embarked', 'Name', 'Cabin', 
                     'PassengerId', 'Ticket', 'Title', 'Deck')
    
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
    data.predictors <- data.useful[, !(names(data.useful) %in% c('Survived'))]
    
    missing <- is.na(data.predictors$Embarked)
    embark_vec <- as.character(data.predictors$Embarked)
    embark_vec[missing] <- 'missing'
    data.predictors$Embarked <- factor(embark_vec)
    
    mids <- mice(data.predictors, m = 15, maxit = 20)
    
    data.predictors.imp <- complete(mids)
    data.imp <- data.frame(data.predictors.imp, Survived = data$Survived)
    
    return(data.imp)
}

data.train.imp <- my_impute(data.train)

# Preprocess
preProc <- preProcess(data.train.imp, method = c('center', 'scale', 'BoxCox'))

data.train.imp.preproc <- predict(preProc, newdata = data.train.imp)

# Build bagged trees on training set.
model.treebag <- train(Survived ~. -Deck, data = data.train.imp, method = 'treebag')

# Build adaboost model (tune with 25 bootstraps)
cl <- makeCluster(detectCores())
registerDoParallel(cl)

trCon <- trainControl(method = "boot", number = 25, allowParallel = TRUE)
model.adaboost <- train(Survived ~. -Deck, data = data.train.imp, method = 'adaboost', 
                   trainControl = trCon)

stopCluster(cl)

# Performance evaluation on validation set

# Impute first
data.validation.imp <- my_impute(data.validation)

# Preprocess
data.validation.imp.preproc <- predict(preProc, newdata = data.validation.imp)

# Bagged Tree
pred.treebag <- predict(model.treebag, newdata = data.validation.imp)
confusionMatrix(pred.treebag, data.validation$Survived)

# Adaboost
pred.adaboost <- predict(model.adaboost, newdata = data.validation.imp)
confusionMatrix(pred.adaboost, data.validation$Survived)
