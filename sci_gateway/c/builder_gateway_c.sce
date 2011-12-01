CURRENT_PATH = strsubst(get_absolute_file_path("builder_gateway_c.sce"), "\", "/");

lib_name = 'svmlib_c';
		 
//files = get_absolute_file_path('builder_gateway_svm.sce')+['svm.cpp',  'svm.h', 'svm_model_scilab.h', 'svm_model_scilab.c', 'read_sparse.c']; //'svmtrain.c', 'svmpredict.c'];

// PutLhsVar managed by user in sci_sum and in sci_sub
// if you do not this variable, PutLhsVar is added
// in gateway generated (default mode in scilab 4.x and 5.x)
WITHOUT_AUTO_PUTLHSVAR = %T;

table = ["libsvmread", "sci_libsvmread"; ...
            "libsvmwrite", "sci_libsvmwrite"; ...
             "svmtrain", "sci_svmtrain";...
               "svmpredict", "sci_svmpredict";...
             "train", "sci_train";...
               "predict", "sci_predict"  ];


files = [ "libsvmread.c","libsvmwrite.c","svm.cpp","svmtrain.c","svm_model_scilab.c","svmpredict.c",..
          "tron.cpp", "linear.cpp", "train.c", "predict.c","linear_model_scilab.c"];

CFLAGS = '-ggdb -I' + CURRENT_PATH;

tbx_build_gateway(lib_name, table, files, CURRENT_PATH, "", "", CFLAGS);


clear lib_name table files  CURRENT_PATH;

clear WITHOUT_AUTO_PUTLHSVAR;