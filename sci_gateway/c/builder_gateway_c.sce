// This file is released under the 3-clause BSD license. See COPYING-BSD.

function builder_gw_c()

CURRENT_PATH = strsubst(get_absolute_file_path("builder_gateway_c.sce"), "\", "/");

lib_name = 'svmlib_c';
		 
// PutLhsVar managed by user in sci_sum and in sci_sub
// if you do not this variable, PutLhsVar is added
// in gateway generated (default mode in scilab 4.x and 5.x)
WITHOUT_AUTO_PUTLHSVAR = %T;

table = ["libsvmread", "sci_libsvmread"; ...
            "libsvmwrite", "sci_libsvmwrite"; ...
             "libsvm_svmtrain", "sci_svmtrain";...
             "libsvm_svmpredict", "sci_svmpredict";...
             "libsvm_lintrain", "sci_train";...
             "libsvm_linpredict", "sci_predict"  ];


files = [ "libsvmread.c","libsvmwrite.c","svm.cpp","svmtrain.c","svm_model_scilab.c","svmpredict.c",..
          "tron.cpp", "linear.cpp", "train.c", "predict.c","linear_model_scilab.c"];

CFLAGS = '-I' + CURRENT_PATH+ " -D__USE_DEPRECATED_STACK_FUNCTIONS__";

tbx_build_gateway(lib_name, table, files, CURRENT_PATH, "", "", CFLAGS);

endfunction

builder_gw_c();
clear builder_gw_c; // remove builder_gw_c on stack