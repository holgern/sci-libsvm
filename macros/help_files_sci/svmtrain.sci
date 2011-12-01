function model = svmtrain(training_label_vector, training_instance_matrix, libsvm_options)
// trains a svm model
// Calling Sequence
// model = svmtrain(training_label_vector, training_instance_matrix);
// model = svmtrain(training_label_vector, training_instance_matrix,libsvm_options);
// Parameters
// libsvm_options:
// s svm_type : set type of SVM (default 0)
// 0 : C-SVC
// 1 : nu-SVC
// 2 : one-class SVM
// 3 : epsilon-SVR
// 4 : nu-SVR
// -t kernel_type : set type of kernel function (default 2)
// 0 -- linear: u'*v
// 1 -- polynomial: (gamma*u'*v + coef0)^degree
// 2 -- radial basis function: exp(-gamma*|u-v|^2)
// 3 -- sigmoid: tanh(gamma*u'*v + coef0)
// 4 -- precomputed kernel: (kernel values in training_instance_matrix)
// -d degree : set degree in kernel function (default 3)
// -g gamma : set gamma in kernel function (default 1/num_features)
// -r coef0 : set coef0 in kernel function (default 0)
// -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
// -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
// -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
// -m cachesize : set cache memory size in MB (default 100)
// -e epsilon : set tolerance of termination criterion (default 0.001)
// -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
// -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
// -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
// -v n : n-fold cross validation mode
// -q : quiet mode (no outputs)
// 
// model structure:
//  model.Parameters: parameters
//  model.nr_class: number of classes; = 2 for regression/one-class svm
//  model.totalSV: total #SV
//  model.rho: -b of the decision function(s) wx+b
//  model.Label: label of each class; empty for regression/one-class SVM
//  model.ProbA: pairwise probability information; empty if -b 0 or in one-class SVM
//  model.ProbB: pairwise probability information; empty if -b 0 or in one-class SVM
//  model.nSV: number of SVs for each class; empty for regression/one-class SVM
//  model.sv_coef: coefficients for SVs in decision functions
//  model.SVs: support vectors
// Description
// The k in the -g option means the number of attributes in the input data.
// 
// option -v randomly splits the data into n parts and calculates crossvalidation accuracy/mean squared error on them.
// 
// Scale your data. For example, scale each attribute to [0,1] or [-1,+1].
// 
// The 'svmtrain' function returns a model which can be used for future
//prediction.  It is a structure and is organized as [Parameters, nr_class,
//totalSV, rho, Label, ProbA, ProbB, nSV, sv_coef, SVs]:
//
// If you do not use the option '-b 1', ProbA and ProbB are empty
//matrices. If the '-v' option is specified, cross validation is
//conducted and the returned model is just a scalar: cross-validation
//accuracy for classification and mean-squared error for regression.
//
//See also
//svmpredict
// Authors
// Chih-Chung Chang
// Chih-Jen Lin
// Holger Nahrstaedt

