# Easy

Run some R code that shows you know how to train and test a decision tree model (rpart, partykit, etc). Bonus points if you can get trtf running for an interval regression problem, for example data(neuroblastomaProcessed, package=”penaltyLearning”). Use 5-fold cross-validation to compare the learned decision tree models to a trivial baseline (which ignores the features and just learns the most likely prediction based on the train labels and always predicts that).

## Classification tree on IRIS dataset - rpart

#### Fitting
```R
tree <- rpart(Species ~ .-Species, data=iris, subset = training_set_index, method = "class",
              control = rpart.control(minbucket = 5, xval = 5))
```
![alt text](https://github.com/abinthomasonline/mmit_gsoc2018/easy/plots/rpart_tree.png "Classification tree")

#### Prediction
```R
predictions <- predict(tree, test_features, type="class")
print(mean(test_labels!=predictions)) #misclassification error
```

```sh
[1] 0.03333333
```

#### Pruning
Pruning reduces the complexity of tree with minimum loss in accuracy. Optimum Complexity Parameter(CP) can be chosen from cpplot.

```R
plotcp(tree)
```

![alt text](https://github.com/abinthomasonline/mmit_gsoc2018/easy/plots/rpart_cp.png "CP plot")

```R
pruned_tree <- prune(tree, 0.1)
```

![alt text](https://github.com/abinthomasonline/mmit_gsoc2018/easy/plots/rpart_pruned.png "Pruned Tree")

```R
predictions_new <- predict(pruned_tree, test_features, type="class")
print(mean(test_labels!=predictions_new)) #misclassification error
```

```sh
[1] 0.03333333
```

#### Trivial Baseline Classifier
It is the tree that always predicts most likely label from training data. That is equivalent to a tree with one leaf.

```R
trivial_tree <- prune(tree, 0.6)
rpart.plot(trivial_tree)
```

![alt text](https://github.com/abinthomasonline/mmit_gsoc2018/easy/plots/rpart_trivial.png "Trivial Tree")

```R
predictions_trivial <- predict(trivial_tree, test_features, type="class")
print(mean(test_labels!=predictions_trivial)) #misclassification error
```

```sh
[1] 0.7333333
```

#### 5-fold cross-validation
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