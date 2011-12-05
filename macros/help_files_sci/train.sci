function model = train(weight_vector, training_label_vector, training_instance_matrix, 'liblinear_options', 'col')
// trains a linear model
// Calling Sequence
// model = train(training_label_vector, training_instance_matrix)
// model = train(training_label_vector, training_instance_matrix, 'liblinear_options')
// model = train(training_label_vector, training_instance_matrix, 'liblinear_options', 'col')
// model = train(weight_vector, training_label_vector, training_instance_matrix, ['liblinear_options', 'col'])
// Parameters
// liblinear_options:
// -s type : set type of solver (default 1)
//  0:  L2-regularized logistic regression (primal)
//  1: L2-regularized L2-loss support vector classification (dual)
//  2: L2-regularized L2-loss support vector classification (primal)
//  3: L2-regularized L1-loss support vector classification (dual)
//  4: multi-class support vector classification by Crammer and Singer
//  5: L1-regularized L2-loss support vector classification
//  6: L1-regularized logistic regression
//  7: L2-regularized logistic regression (dual)
//  -c cost : set the parameter C (default 1)
//  -e epsilon : set tolerance of termination criterion	
//  -s 0 and 2:  |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2
//  -s 1, 3, 4 and 7: Dual maximal violation <= eps; similar to libsvm (default 0.1)
//  -s 5 and 6: |f'(w)|_inf <= eps*min(pos,neg)/l*|f'(w0)|_inf, where f is the primal function (default 0.01)
// -B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)
// -wi weight: weights adjust the parameter C of different classes (see README for details)
// -v n: n-fold cross validation mode
// -q : quiet mode (no outputs)
// col: if 'col' is setted, training_instance_matrix is parsed in column format, otherwise is in row format
// 
// model Structure:
// model.Parameters: Parameters
// model.nr_class: number of classes
// model.nr_feature: number of features in training data (without including the bias term)
// model.bias: If >= 0, we assume one additional feature is added to the end         of each data instance.
// model.Label: label of each class
// model.w: a nr_w-by-n matrix for the weights, where n is nr_feature      or nr_feature+1 depending on the existence of the bias term.          nr_w is 1 if nr_class=2 and -s is not 4 (i.e., not         multi-class svm by Crammer and Singer). It is          nr_class otherwise.
// 
// Description
// The 'train' function returns a model which can be used for future prediction.  It is a structure and is organized as [Parameters, nr_class, nr_feature, bias, Label, w]
// 
// If the '-v' option is specified, cross validation is conducted and the returned model is just a scalar: cross-validation accuracy.
// Examples
// label_vector=[zeros(20,1);ones(20,1)];
// instance_matrix = sparse([rand(20,2); -1*rand(20,2)]);
// model=train(label_vector,instance_matrix,"-q")
// [pred_label, accuracy, dec_values]=predict(label_vector,instance_matrix,model)
// See also
// predict
// Authors
// Chih-Chung Chang
// Chih-Jen Lin
// Holger Nahrstaedt

