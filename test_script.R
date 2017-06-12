data <- read.csv('train.csv', header = TRUE)

# Categorical variables: Sex, Survived, Pclass, Sex, SibSp, 

# response = Survived

n <- nrow(data)

# 80%, 20% training/testing split
inTrain <- sample(1:n, size = ceiling(n*0.8))

data.train <- data[inTrain,]
data.test <- data[-inTrain,]

# Uselss features: Name, Cabin, PassengerId