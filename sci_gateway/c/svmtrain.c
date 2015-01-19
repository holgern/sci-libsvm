//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2000-2011 Chih-Chung Chang and Chih-Jen Lin
// Modifiered 2011 by Holger Nahrstaedt
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither name of copyright holders nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
//CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "svm.h"

#include <api_scilab.h>
// #define __USE_DEPRECATED_STACK_FUNCTIONS__
// #include <stack-c.h>
#include <sciprint.h>
#include <MALLOC.h>
#include <Scierror.h>

#include "svm_model_scilab.h"

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

//#define DEBUG

#define NUM_OF_RETURN_FIELD 10
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))




void svm_print_null(const char *s) {}
void svm_print_string_scilab(const char *s) {sciprint(s);}
static void (*svmtrain_print_string)(const char *) = svm_print_string_scilab;
void exit_with_help_train()
{
        Scierror (999,
        "Usage: model = libsvm_svmtrain(training_label_vector, training_instance_matrix, 'libsvm_options');\n"
	"libsvm_options:\n"
	"-s svm_type : set type of SVM (default 0)\n"
	"	0 -- C-SVC		(multi-class classification)\n"
	"	1 -- nu-SVC		(multi-class classification)\n"
	"	2 -- one-class SVM\n"
	"	3 -- epsilon-SVR	(regression)\n"
	"	4 -- nu-SVR		(regression)\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_instance_matrix)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight : set the parameter C of class i to weight*C, for C-SVC  and nu-SVC (default 1)\n"
	"-v n : n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
}

// svm arguments
struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int is_cross_validation; // has to be renamed to is_cross_validation as scilab crashes otherwise
int nr_fold;
int max_index;

/* MAD begin changes  */
/*double do_cross_validation()  */
void svm_do_cross_validation(double *ptr)
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);
	double retval = 0.0;
  char buffer [100];

	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
	//#ifdef DEBUG
  sprintf(buffer,"Cross Validation Mean squared error = %g\n",total_error/prob.l);
  svmtrain_print_string(buffer);
  sprintf(buffer,"Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
      svmtrain_print_string(buffer);
	//#endif
		retval = total_error/prob.l;
    ptr[0] = retval;
    ptr[1] = 0.0;
    ptr[2] = 0.0;
	}
	else
	{
    int countp = 0;
    int countn = 0;
    int correctp = 0;
    int correctn = 0;

		for(i=0;i<prob.l;i++)
		 	if(target[i] == prob.y[i])
		 		++total_correct;

    for(i=0;i<prob.l;i++)
    {
      if(prob.y[i] > 0.0)
      {
        countp++;
        if(target[i] == prob.y[i])
          correctp++;
        }
        else
        {
          countn++;
          if(target[i] == prob.y[i])
            correctn++;
          }
        }

	//#ifdef DEBUG

  sprintf(buffer,"Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
  svmtrain_print_string(buffer);
  sprintf(buffer,"Positive Cross Validation Accuracy = %g%%\n",100.0*correctp/countp);
  svmtrain_print_string(buffer);
  sprintf(buffer,"Negative Cross Validation Accuracy = %g%%\n",100.0*correctn/countn);
  svmtrain_print_string(buffer);
    //    #endif
		retval = 100.0*total_correct/prob.l;
    ptr[0] = retval;
    retval = 100.0*correctp/countp;
    ptr[1] = retval;
    retval = 100.0*correctn/countn;
    ptr[2] = retval;
	}
	free(target);
	//return retval;
}
/* MAD end changes */

// nrhs should be 3
int svm_parse_command_line(int nrhs, char *cmd, char *model_file_name)
{
	int i, argc = 1;
  //char cmd[CMD_LEN];
	char *argv[CMD_LEN/2];
  void (*print_func)(const char *) = svm_print_string_scilab;        // default printing to stdout
	int m1 = 0, n1 = 0;

	// default values
  svmtrain_print_string =  &svm_print_string_scilab;
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	is_cross_validation = 0;

	if(nrhs <= 1)
		return 1;

	if(nrhs > 2)
	{

		if((argv[argc] = strtok(cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;

		//}
		//}
	}

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		if(i>=argc && argv[i-1][1] != 'q')	// since option -q has no parameter
			return 1;
		switch(argv[i-1][1])
		{
			case 's':
				param.svm_type = atoi(argv[i]);
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'q':
                                print_func = &svm_print_null;
                                svmtrain_print_string =  &svm_print_null;
				i--;
				break;
			case 'v':
				is_cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					Scierror (999,"n-fold cross validation: n must >= 2\n");
					return 1;
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			default:
				Scierror (999,"Unknown option -%c\n", argv[i-1][1]);
				return 1;
		}
	}

	svm_set_print_string_function(print_func);

	return 0;
}

// read in a problem (in svmlight format)
int svm_read_problem_dense(int *label_vec, int *instance_mat)
{
        int i, j, k, l ,r_samples, c_samples, r_labels, c_labels, index;
        int elements, sc, label_vector_row_num;
	double *samples = NULL, *labels = NULL;
	SciErr _SciErr;

	prob.x = NULL;
	prob.y = NULL;
	x_space = NULL;


        _SciErr = getMatrixOfDouble(pvApiCtx, label_vec, &r_labels, &c_labels, &labels);
	if(_SciErr.iErr)
		{
			printError(&_SciErr, 0);
			return -1;
		}
	if (c_labels > 1){
	  Scierror (999,"Error: Wrong size for argument %d: Row Vector expected.\n", 1);
	  return -1;
	}
	  if (c_labels*r_labels == 0){
	    Scierror (999,"Error: Wrong size for input argument #%d: Non-empty vector expected.\n", 1);
	    return -1;
	  }
        _SciErr = getMatrixOfDouble(pvApiCtx, instance_mat, &r_samples, &c_samples, &samples);
	if(_SciErr.iErr)
		{
			printError(&_SciErr, 0);
			return -1;
		}
	if (c_samples*r_samples == 0){
	    Scierror (999,"Error: Wrong size for input argument #%d: Non-empty matrix expected.\n", 2);
	    return -1;
	  }

	//labels = mxGetPr(label_vec);
	//samples = mxGetPr(instance_mat);
	//sc = (int)mxGetN(instance_mat);
	sc = c_samples;

	elements = 0;
	// the number of instance
	l = r_samples;//(int)mxGetM(instance_mat);
	label_vector_row_num = r_labels;//(int)mxGetM(label_vec);
	prob.l = (int)l;

	if(label_vector_row_num!=l)
	{
		Scierror(999,"Length of label vector does not match # of instances.\n");
		return -1;
	}

	if(param.kernel_type == PRECOMPUTED)
		elements = l * (sc + 1);
	else
	{
		for(i = 0; i < l; i++)
		{
			for(k = 0; k < sc; k++)
				if(samples[k * l + i] != 0)
					elements++;
			// count the '-1' element
			elements++;
		}
	}

	prob.y = Malloc(double,l);
	prob.x = Malloc(struct svm_node *,l);
	x_space = Malloc(struct svm_node, elements);

	max_index = sc;
	j = 0;
	for(i = 0; i < l; i++)
	{
		prob.x[i] = &x_space[j];
		prob.y[i] = labels[i];

		for(k = 0; k < sc; k++)
		{
			if(param.kernel_type == PRECOMPUTED || samples[k * l + i] != 0)
			{
				x_space[j].index = (int)k + 1;
				x_space[j].value = samples[k * l + i];
				j++;
			}
		}
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = (double)(1.0/max_index);

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<l;i++)
		{
			if((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > (int)max_index)
			{
				Scierror(999,"Wrong input format: sample_serial_number out of range\n");
				return -1;
			}
		}

	return 0;
}

int svm_read_problem_sparse(int *label_vec,  int *instance_mat)
{
        int i, j,jj, k, l, low, high,r_labels, c_labels,r_samples, c_samples;
        int *ir, *jc;
        int elements, num_samples, label_vector_row_num;
	double *samples, *labels;
        int *instance_mat_col; // transposed instance sparse matrix
        SciErr _SciErr;

	prob.x = NULL;
	prob.y = NULL;
	x_space = NULL;

	// transpose instance matrix
//         {
//                 mxArray *prhs[1], *plhs[1];
//                 prhs[0] = mxDuplicateArray(instance_mat);
//                 if(mexCallSCILAB(1, plhs, 1, prhs, "transpose"))
//                 {
//                         sciprint("Error: cannot transpose training instance matrix\n");
//                         return -1;
// 		}
// 		instance_mat_col = plhs[0];
// 		mxDestroyArray(prhs[0]);
// 	}

	 _SciErr = getMatrixOfDouble(pvApiCtx, label_vec, &r_labels, &c_labels, &labels);
	 if(_SciErr.iErr)
	{
			printError(&_SciErr, 0);
			return -1;
		}

	if (c_labels > 1){
	  Scierror (999,"Error: Wrong size for argument %d: Row Vector expected.\n", 1);
	  return -1;
	}
	  if (c_labels*r_labels == 0){
	    Scierror (999,"Error: Wrong size for input argument #%d: Non-empty vector expected.\n", 1);
	    return -1;
	  }
	 _SciErr = getSparseMatrix(pvApiCtx,instance_mat,&r_samples, &c_samples, &num_samples, &ir, &jc, &samples);
	 if(_SciErr.iErr)
		{
			printError(&_SciErr, 0);
			return -1;
		}
      if (c_samples*r_samples == 0){
	    Scierror (999,"Error: Wrong size for input argument #%d: Non-empty matrix expected.\n", 2);
	    return -1;
	}

	// each column is one instance
	//labels = mxGetPr(label_vec);
	//samples = mxGetPr(instance_mat_col);
	//ir = mxGetIr(instance_mat_col);
	//jc = mxGetJc(instance_mat_col);

	//num_samples = (int)mxGetNzmax(instance_mat_col);

	// the number of instance
	//prob.l = (int)mxGetN(instance_mat_col);
	l = r_samples;
	label_vector_row_num = r_labels;//(int)mxGetM(label_vec);
	prob.l = (int) l;

	if(label_vector_row_num!=l)
	{
		Scierror (999,"Length of label vector does not match # of instances.\n");
		return -1;
	}

	elements = num_samples + l;
	//max_index = (int)mxGetM(instance_mat_col);
	max_index = c_samples;
#ifdef DEBUG
        printf("prob.l %d, label_vector_row_num %d, elements %d, max_index %d\n",l,label_vector_row_num,elements,max_index);
#endif
	prob.y = Malloc(double,l);
	prob.x = Malloc(struct svm_node *,l);
	x_space = Malloc(struct svm_node, elements);

	j = 0;jj=0;
	for(i=0;i<l;i++)
	{
		prob.x[i] = &x_space[j];
		prob.y[i] = labels[i];
		//low = (int)jc[i];
		low = 0;
		high = (int)ir[i];
		for(k=low;k<high;k++)
		{
			x_space[j].index = (int)jc[jj];
			x_space[j].value = samples[jj];
			j++;jj++;
	 	}
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = (double)(1.0/max_index);

	return 0;
}


// Interface function of scilab
// now assume prhs[0]: label prhs[1]: features

int sci_svmtrain(char * fname)

{
         SciErr _SciErr;
	const char *error_msg;
	int * p_label_vector = NULL;
        int * p_instance_matrix = NULL;
	int * p_option_string = NULL;
	int r_samples, c_samples;
	double *samples = NULL;
        int type,type3;
	char * option_string = NULL;
	// fix random seed to have same results for each run
	// (for cross validation and probability estimation)
	srand(1);

	// Transform the input Matrix to libsvm format
        if(nbInputArgument(pvApiCtx) > 1 && nbInputArgument(pvApiCtx) < 4)
	{
		int err;

		_SciErr = getVarAddressFromPosition(pvApiCtx, 1, &p_label_vector);
		if(_SciErr.iErr)
		{
			printError(&_SciErr, 0);
			return 0;
		}
                _SciErr = getVarType(pvApiCtx, p_label_vector, &type);
		if(_SciErr.iErr)
		{
			printError(&_SciErr, 0);
			return 0;
		}
		if (type!=sci_matrix && type!=sci_sparse)
		{
		  Scierror (999,"%s: label vector must be double\n",fname);
		  return 0;
		}
		_SciErr = getVarAddressFromPosition(pvApiCtx, 2, &p_instance_matrix);
		if(_SciErr.iErr)
		{
			printError(&_SciErr, 0);
			return 0;
		}
                _SciErr = getVarType(pvApiCtx, p_instance_matrix, &type);
		if(_SciErr.iErr)
		{
			printError(&_SciErr, 0);
			return 0;
		}
		 if (type!=sci_matrix && type!=sci_sparse)
		{
		  Scierror (999,"%s: instance matrix must be double\n",fname);
		  return 0;
		}

		if (nbInputArgument(pvApiCtx)==3) {
		    _SciErr = getVarAddressFromPosition(pvApiCtx, 3, &p_option_string);
		    if(_SciErr.iErr)
		{
			    printError(&_SciErr, 0);
			    return 0;
		    }
		    _SciErr = getVarType(pvApiCtx, p_option_string, &type3);
		    if(_SciErr.iErr)
		    {
			    printError(&_SciErr, 0);
			    return 0;
		    }
		 if (type3==sci_strings)
		  {
		    getAllocatedSingleString(pvApiCtx, p_option_string, &option_string);
		}


		}

		    if(svm_parse_command_line(nbInputArgument(pvApiCtx), option_string, NULL))
		{
			    exit_with_help_train();
			svm_destroy_param(&param);
			    return 0;
		}
		  if (option_string != NULL)
		    freeAllocatedSingleString(option_string);


		if(type==sci_sparse)
		{

			if(param.kernel_type == PRECOMPUTED)
			{
			   Scierror (999,"%s: Precomputed kernel requires dense matrix\n",fname);
		            return 0;
			  /*
				// precomputed kernel requires dense matrix, so we make one
				mxArray *rhs[1], *lhs[1];

				rhs[0] = mxDuplicateArray(prhs[1]);
                                if(mexCallSCILAB(1, lhs, 1, rhs, "full"))
				{
                                        sciprint("Error: cannot generate a full training instance matrix\n");
					svm_destroy_param(&param);
                                        svm_fake_answer();
					return;
				}
                                err = svm_read_problem_dense(prhs[0], lhs[0]);
				mxDestroyArray(lhs[0]);
                                mxDestroyArray(rhs[0]);*/
			}
			else
                                err = svm_read_problem_sparse(p_label_vector, p_instance_matrix);

		}
		else
                        err = svm_read_problem_dense(p_label_vector, p_instance_matrix);
		#ifdef DEBUG
                    printf("DEBUG: read problem done\n");
                 #endif

		// svmtrain's original code
		error_msg = svm_check_parameter(&prob, &param);
		#ifdef DEBUG
                    printf("DEBUG: check parameter done\n");
                 #endif
		if(err || error_msg)
		{
			if (error_msg != NULL)
				Scierror (999,"%s: %s\n", fname,error_msg);
			else
				Scierror(999,"Error!\n");
			svm_destroy_param(&param);
			free(prob.y);
			free(prob.x);
			free(x_space);
                        return 0;
		}

		if(is_cross_validation)
		{
			double *ptr;
			_SciErr = allocMatrixOfDouble(pvApiCtx, nbInputArgument(pvApiCtx) + 1, 3, 1, &ptr);
			if(_SciErr.iErr)
		  {
			    printError(&_SciErr, 0);
			    return 0;
		  }

      /* MAD begin changes */
      //double *ptr;
      //plhs[0] = mxCreateDoubleMatrix(2, 1, mxREAL);
      //ptr = mxGetPr(plhs[0]);
      /*ptr[0] = svm_do_cross_validation(); */
      svm_do_cross_validation(ptr);
      /* MAD end changes */

			AssignOutputVariable(pvApiCtx,1) = nbInputArgument(pvApiCtx) + 1;
			  /* This function put on scilab stack, the lhs variable
			which are at the position lhs(i) on calling stack */
			/* You need to add PutLhsVar here because WITHOUT_ADD_PUTLHSVAR
			was defined and equal to %t */
			/* without this, you do not need to add PutLhsVar here */
			ReturnArguments(pvApiCtx);
		}
		else
		{
		   //_SciErr = getMatrixOfDouble(pvApiCtx, p_instance_matrix, &r_samples, &c_samples, &samples);

                        int nr_feat = max_index;//mxGetN(prhs[1]);
                        #ifdef DEBUG
                           printf("c_samples %d\n",nr_feat);
                        #endif
                        //const char *error_msg;
			model = svm_train(&prob, &param);
			#ifdef DEBUG
                              printf("DEBUG: svm train done\n");
                        #endif
                        _SciErr = model_to_scilab_structure(nr_feat, model);
			if(_SciErr.iErr)
		        {
			    printError(&_SciErr, 0);
			    exit_with_help_train();
			    return 0;
		         }
			 AssignOutputVariable(pvApiCtx,1) = nbInputArgument(pvApiCtx)+1;
		      // AssignOutputVariable(pvApiCtx,2) = nbInputArgument(pvApiCtx)+2;
		  /* This function put on scilab stack, the lhs variable
		  which are at the position lhs(i) on calling stack */
		  /* You need to add PutLhsVar here because WITHOUT_ADD_PUTLHSVAR
		  was defined and equal to %t */
		  /* without this, you do not need to add PutLhsVar here */
		      ReturnArguments(pvApiCtx);
                        //if(error_msg)
                        //        sciprint("Error: can't convert libsvm model to matrix structure: %s\n", error_msg);
			svm_free_and_destroy_model(&model);
		}
		svm_destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
	}
	else
	{
                exit_with_help_train();
                return 0;
	}
}
