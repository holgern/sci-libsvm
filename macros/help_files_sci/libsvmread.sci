function [label_vector, instance_matrix] = libsvmread('data.txt'); 
//   reads files in LIBSVM format
// Calling Sequence
//	[label_vector, instance_matrix] = libsvmread('data.txt'); 
// Description
// Two outputs are labels and instances, which can then be used as inputs
// of svmtrain or svmpredict. 
// 
// 
// Examples
// [heart_scale_label, heart_scale_inst] = libsvmread('heart_scale');
// See also
// libsvmwrite
// Authors
// Chih-Chung Chang
// Chih-Jen Lin
// Holger Nahrstaedt
