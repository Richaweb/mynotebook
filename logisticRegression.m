%% :Logistic Regression
% 
%  This file contains following functions code that helps you to build model on the logistic
%  regression.
%     sigmoid.m
%     costFunction.m
%     predict.m

%% Initialization
clear ; close all; clc

%%  ============ Part 1: Read Data ============

data = csvread('Breastcancer_New.csv');
X = data(:, [2:end]); 
y = data(:, 1);

%% ============ Part 2: Compute Cost and Gradient ============

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];
Xtrain=X([1:470],:)
ytrain=y([1:470],:)
Xtest=X([471:end],:)
ytest=y([471:end],:)

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);
% Compute and display  cost and gradient
iter=1000; % No. of iterations for weight updation
alpha=0.1 % Learning parameter 
[J grad h theta]=costFunction(initial_theta,Xtrain,ytrain,alpha,iter) % Cost funtion

%% ============== Part 3: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data.  
ytrainpred = double(predict(theta, Xtrain));
ytestpred = double(predict(theta, Xtest));
% Compute accuracy on our training set

confusion_matrix = confusionmat(ytrain,ytrainpred)
accuracy=trace(confusion_matrix)/(sum(sum(confusion_matrix)))
confusion_matrix = confusionmat(ytest,ytestpred)
accuracy=trace(confusion_matrix)/(sum(sum(confusion_matrix)))
