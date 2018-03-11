#LOGISTIC REGRESSION MODEL TREE ON TITANIC SURVIVAL DATASET
library(partykit)
library(vcd)
library(grid)
library(libcoin)
library(mvtnorm)
library(rpart)

set.seed(321)

data("Titanic")

#data transformation
titanic_df <- as.data.frame(Titanic)
titanic_df <- titanic_df[rep(1:nrow(titanic_df), titanic_df$Freq), 1:4]
titanic_df <- transform(titanic_df, MaleAdult = factor(Sex=="Female"|Age=="Child"))

training_set_index <- c(sample(1:nrow(titanic_df), 1800))

#fitting
tree <- glmtree(Survived ~ MaleAdult | Class + Sex + Age, 
                data = titanic_df, subset = training_set_index, 
                family = binomial, alpha=0.01, catsplit="multiway")
plot(tree)

test_features <- titanic_df[-training_set_index, names(titanic_df)!="Survived"]
test_labels <- titanic_df[-training_set_index, "Survived"]

#prediction
predictions <- predict(tree, test_features, type = "response")
predictions <- ifelse(predictions>=0.5, "Yes", "No")
print(mean(test_labels!=predictions)) #misclassification error

#trivial baseline
mode <- function(x) {
  unique_x <- na.omit(unique(x))
  unique_x[which.max(tabulate(match(x, unique_x)))]
}
trivial_predictions <- rep(mode(titanic_df[training_set_index, "Survived"]), length(test_labels))
print(mean(test_labels!=trivial_predictions)) #misclassification error

#cross-validation
titanic_df_cv <- titanic_df[training_set_index,]
n_folds <- 5
subgroup_index <- sample(rep(1:n_folds, length(titanic_df_cv)))
tree_missclassification_error <- rep(0,5)
trivial_missclassification_error <- rep(0,5)
for (i in 1:n_folds) {
  training_set_index <- which(subgroup_index!=i)
  test_features <- titanic_df_cv[-training_set_index, names(titanic_df_cv)!="Survived"]
  test_labels <- titanic_df_cv[-training_set_index, "Survived"]
  
  #tree
  tree <- glmtree(Survived ~ MaleAdult | Class + Sex + Age, 
                  data = titanic_df_cv, subset = training_set_index, 
                  family = binomial, alpha=0.01, catsplit="multiway")
  predictions <- predict(tree, test_features, type = "response")
  predictions <- ifelse(predictions>=0.5, "Yes", "No")
  tree_missclassification_error[i] <- mean(test_labels!=predictions)
  
  #trivial
  trivial_predictions <- rep(mode(titanic_df_cv[training_set_index, "Survived"]), length(test_labels))
  trivial_missclassification_error[i] <- mean(test_labels!=trivial_predictions)
}
print(mean(tree_missclassification_error))
print(mean(trivial_missclassification_error))

