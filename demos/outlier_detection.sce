
mode(1);lines(0);


    
// Welcome to the libsvm Toolbox for Scilab 
// This demonstration  shows how one class classification can be used for outlier detection
//
// Press any key to continue...
 
halt('Press return'); clc;
//
// We are generating a random dataset and add one outlier
x=rand(90,1);
x(10)=1.5;

// Press any key to continue...
halt('Press return');
//
// At first we will train and test the dataset. We set the classifier to one-class SVM (-s 2)
model=svmtrain(ones(x),x,'-s 2');
//
[label,acc,dec_val]=svmpredict(ones(x),x,model);
//

// Now we can check the decision_values
// Press any key to continue...
halt('Press return');

scf();clf();
plot(dec_val);
xgrid(1);
xtitle("The is an outlier at pos 10");


