#CLASSIFICATION TREE ON IRIS DATASET
library(rpart)
library(rpart.plot)

set.seed(123)

data("iris")

training_set_index <- c(sample(1:nrow(iris), 120))

#fitting
tree <- rpart(Species ~ .-Species, data=iris, subset = training_set_index, method = "class",
              control = rpart.control(minbucket = 5, xval = 5))
rpart.plot(tree)

test_features <- iris[-training_set_index, names(iris)!="Species"]
test_labels <- iris[-training_set_index, "Species"]

#prediction
predictions <- predict(tree, test_features, type="class")
print(mean(test_labels!=predictions)) #misclassification error

#pruning
plotcp(tree)
pruned_tree <- prune(tree, 0.1)
rpart.plot(pruned_tree)

#prediction_new
predictions_new <- predict(pruned_tree, test_features, type="class")
print(mean(test_labels!=predictions_new)) #misclassification error

#trivial baseline classifier
trivial_tree <- prune(tree, 0.6)
rpart.plot(trivial_tree)

#prediction_trivial
predictions_trivial <- predict(trivial_tree, test_features, type="class")
print(mean(test_labels!=predictions_trivial)) #misclassification error

#cptable
print(pruned_tree$cptable)
print(trivial_tree$cptable)