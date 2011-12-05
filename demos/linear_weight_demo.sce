demopath = get_absolute_file_path("linear_weight_demo.sce");


mode(1);lines(0);


    
// Welcome to the libsvm Toolbox for Scilab 
// This demonstration  shows LIBLINEAR with instance weight support
//
// Press any key to continue...
 
halt('Press return'); clc;

//Train and test on the provided data heart_scale
    
[heart_scale_label, heart_scale_inst] = libsvmread(demopath+'/heart_scale');

heart_scale_weight = fscanfMat(demopath+'/heart_scale.wgt');

model = train(heart_scale_weight, heart_scale_label, heart_scale_inst, '-c 1 -q');

// test the training data
// 
[predict_label, accuracy, dec_values] = predict(heart_scale_label, heart_scale_inst, model); 
//
//
disp("Accuracy: "+string(accuracy(1)));
//
//
// Press any key to continue...
halt('Press return'); 

//Train and test without weights:

 model = train(heart_scale_label, heart_scale_inst, '-c 1 -q');
// test the training data
[predict_label, accuracy, dec_values] = predict(heart_scale_label, heart_scale_inst, model); 
//
//
disp("Accuracy: "+string(accuracy(1)));
