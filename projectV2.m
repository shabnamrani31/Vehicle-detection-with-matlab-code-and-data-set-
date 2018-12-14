% Team Owais, Zerk, Shaleem, Faisal, Farman
% Accuracy 0.9472 (94%)
% Uses bag of features for training.

% Path to data-set folder
path = 'data-set';
imgSets = imageSet(path, 'recursive');

% Partition our data to 30/70, Where 30% data is for training and 70 for
% test set. Increasing training set will increase accuracy.
[trainingSets, testSets] = partition(imgSets, 0.7, 'randomize'); 

% Extract features from training set images.
bag = bagOfFeatures(trainingSets,'Verbose',true,'StrongestFeatures', 0.8, 'GridStep', [8 8]);

% Generate category classifier from training images.
categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);

% Test classifier on test image.
confMatrix = evaluate(categoryClassifier, testSets)

% Show mean of diagnols, Diagnol contains success.
mean(diag(confMatrix))

% Test classifier against our random input
img = imread('nv.png');
[labelIdx, score] = predict(categoryClassifier, img);

% Result: Non Vehicle -> Success
categoryClassifier.Labels(labelIdx)