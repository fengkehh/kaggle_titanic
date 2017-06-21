library('lattice')
library('caret')
library('missForest')
library('parallel')
library('doParallel')

## functions

# Data PreProcessing
# Function that checks if the measurement has no missing values for useful features.

chk_missing <- function(datapoint, useful) {
    useful_vars <- names(datapoint)[names(datapoint) %in% useful]
    
    return(sum(is.na(datapoint[useful_vars])) > 0)
}


# Remove data with missing values and return preprocessed data, chosen indices 
# and original data.
my_preProcess <- function(data, useful) {
    n <- nrow(data)
    
    good_ind <- rep(FALSE, n)
    
    for (i in 1:n) {
        good_ind[i] = !chk_missing(data[i,], useful)
    }
    
    ans = list(data = data[good_ind,], ind = good_ind, original = data)
    
    return(ans)
}


# Data imputation
my_impute <- function(data, useful) {
    imputed_vars <- missForest(data[,names(data) %in% useful], ntree = 5001, 
                               variablewise = TRUE)
    
    data[,names(data) %in% useful] <- imputed_vars$ximp
    
    ans = list(data.imp = data, obj.imp = imputed_vars)
    return(ans)
}

my_predict <- function(train.model, data, useful) {
    # Preprocess testing set.
    data.preProc <- my_preProcess(data, useful)
    
    pred <- rep(NA, nrow(data))
    
    pred[data.preProc$ind] <- as.numeric(as.character(predict(train.model, 
                                                              data.preProc$data)))
    pred <- factor(pred)
    
    return(pred)
}