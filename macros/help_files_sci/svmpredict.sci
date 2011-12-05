function [predicted_label, accuracy, decision_values/prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')
// Does prediction for a calculated svm model
// Calling Sequence
// [predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector, testing_instance_matrix, model)
// [predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')
// Parameters
//  model: SVM model structure from svmtrain.
//   libsvm_options:-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet
//   predicted_label: SVM prediction output vector
//    accuracy: a vector with accuracy, mean squared error, squared correlation coefficient.
//    prob_estimates: If selected, probability estimate vector
//    predicted_label: vector of predicted labels
//    accuracy: a vector including accuracy (for classification), mean squared error, and squared correlation coefficient (for regression).
//    decision_values:  a matrix containing decision values or probability estimates (if '-b 1' is specified).
//    
//   Description
// 
//The third output is a matrix containing decision values or probability estimates (if '-b 1' is specified). If k is the number of classes
//in training data, for decision values, each row includes results of predicting k(k-1)/2 binary-class SVMs. 
//
//For classification, k = 1 is a special case. Decision value +1 is returned for each testing instance,instead of an empty vector. 
//
//For probabilities, each row contains k values indicating the probability that the testing instance is in each class.
//
//Note that the order of classes here is the same as 'Label' field in the model structure.
//
// Examples
// label_vector=[zeros(20,1);ones(20,1)];
// instance_matrix = [rand(20,2); -1*rand(20,2)];
// model=svmtrain(label_vector,instance_matrix);
// 
// [pred,acc,dec]=svmpredict(label_vector,instance_matrix,model);
//
// See also
// svmtrain
// Authors
// Chih-Chung Chang
// Chih-Jen Lin
// Holger Nahrstaedt
// 

