# Easy

Run some R code that shows you know how to train and test a decision tree model (rpart, partykit, etc). Bonus points if you can get trtf running for an interval regression problem, for example data(neuroblastomaProcessed, package=”penaltyLearning”). Use 5-fold cross-validation to compare the learned decision tree models to a trivial baseline (which ignores the features and just learns the most likely prediction based on the train labels and always predicts that).

## Classification tree on IRIS dataset - rpart

Tree predicts the Species from petal and sepal dimensions.

### Fitting
```R
tree <- rpart(Species ~ .-Species, data=iris, subset = training_set_index, method = "class",
              control = rpart.control(minbucket = 5, xval = 5))
```
![alt text](https://github.com/abinthomasonline/mmit_gsoc2018/blob/master/easy/plots/rpart_tree.png "Classification tree")

### Prediction
```R
predictions <- predict(tree, test_features, type="class")
print(mean(test_labels!=predictions)) #misclassification error
```

```sh
[1] 0.03333333
```

### Pruning
Pruning reduces the complexity of tree with minimum loss in accuracy. Optimum Complexity Parameter(CP) can be chosen from cpplot.

```R
plotcp(tree)
```

![alt text](https://github.com/abinthomasonline/mmit_gsoc2018/blob/master/easy/plots/rpart_cp.png "CP plot")

```R
pruned_tree <- prune(tree, 0.1)
```

![alt text](https://github.com/abinthomasonline/mmit_gsoc2018/blob/master/easy/plots/rpart_pruned.png "Pruned Tree")

```R
predictions_new <- predict(pruned_tree, test_features, type="class")
print(mean(test_labels!=predictions_new)) #misclassification error
```

```sh
[1] 0.03333333
```

### Trivial Baseline Classifier
It is the tree that always predicts most likely label from training data. That is equivalent to a tree with one leaf.

```R
trivial_tree <- prune(tree, 0.6)
rpart.plot(trivial_tree)
```

![alt text](https://github.com/abinthomasonline/mmit_gsoc2018/blob/master/easy/plots/rpart_trivial.png "Trivial Tree")

```R
predictions_trivial <- predict(trivial_tree, test_features, type="class")
print(mean(test_labels!=predictions_trivial)) #misclassification error
```

```sh
[1] 0.7333333
```

### 5-fold cross-validation
`xval` sets the cross-validation parameter. Risk estimates and standard deviations corresponding to each splits(or CP) are stored in `tree$cptable`.

```R
print(pruned_tree$cptable)
```

```sh
         CP nsplit  rel error    xerror       xstd
1 0.5256410      0 1.00000000 1.1025641 0.06328539
2 0.4102564      1 0.47435897 0.5641026 0.06767806
3 0.1000000      2 0.06410256 0.1025641 0.03503231
```

```R
print(trivial_tree$cptable)
```

```sh
   CP nsplit rel error   xerror       xstd
1 0.6      0         1 1.102564 0.06328539
```

## Model tree (logistic regression) on Titanic dataset - partykit (glmtree)

Tree predicts whether the probability of survival of a male adult given the class. Logistic regression models are formed for each terminal nodes of the tree. Muiltiway splits are allowed.

### Data transformation
```R
titanic_df <- as.data.frame(Titanic)
titanic_df <- titanic_df[rep(1:nrow(titanic_df), titanic_df$Freq), 1:4]
titanic_df <- transform(titanic_df, MaleAdult = factor(Sex=="Female"|Age=="Child"))
```

### Fitting
```R
tree <- glmtree(Survived ~ MaleAdult | Class + Sex + Age, 
                data = titanic_df, subset = training_set_index, 
                family = binomial, alpha=0.01, catsplit="multiway")
```              
![alt text](https://github.com/abinthomasonline/mmit_gsoc2018/blob/master/easy/plots/partykit_tree.png "Classification tree")

### Prediction
```R
predictions <- predict(tree, test_features, type = "response")
predictions <- ifelse(predictions>=0.5, "Yes", "No")
print(mean(test_labels!=predictions)) #misclassification error
```

```sh
[1] 0.1895262
```

### Trivial Baseline Classifier
```R
mode <- function(x) {
  unique_x <- na.omit(unique(x))
  unique_x[which.max(tabulate(match(x, unique_x)))]
}
trivial_predictions <- rep(mode(titanic_df[training_set_index, "Survived"]), length(test_labels))
print(mean(test_labels!=trivial_predictions)) #misclassification error
```

```sh
[1] 0.3491272
```

### 5-fold Cross-Validation
```R
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
```

```R
print(mean(tree_missclassification_error))
print(mean(trivial_missclassification_error))
```

```sh
[1] 0.2482022
[1] 0.3180899
```