function [predicted_label, accuracy, decision_values/prob_estimates] = predict(testing_label_vector, testing_instance_matrix, model, 'liblinear_options','col')
// Does prediction for a calculated svm model
// Calling Sequence
// [predicted_label, accuracy, decision_values/prob_estimates] = predict(testing_label_vector, testing_instance_matrix, model)
// [predicted_label, accuracy, decision_values/prob_estimates] = predict(testing_label_vector, testing_instance_matrix, model,'liblinear_options')
// [predicted_label, accuracy, decision_values/prob_estimates] = predict(testing_label_vector, testing_instance_matrix, model, 'liblinear_options','col')
// Parameters
// liblinear_options: -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0)
// col: if 'col' is setted testing_instance_matrix is parsed in column format, otherwise is in row format
// predicted_label:a vector of predicted labels
// accuracy: a scalar meaning accuracy
// decision_values/prob_estimates: a matrix containing decision values or probability estimates (if '-b 1' is specified).
// Description
// 
//The third output is a matrix containing decision values or probability
//estimates (if '-b 1' is specified). If k is the number of classes
//and k' is the number of classifiers (k'=1 if k=2, otherwise k'=k), for decision values,
//each row includes results of k' binary linear classifiers. For probabilities,
//each row contains k values indicating the probability that the testing instance is in
//each class. Note that the order of classes here is the same as 'Label'
//field in the model structure.
//
//See also
//train
//