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
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "linear.h"

#include <api_scilab.h>
#define __USE_DEPRECATED_STACK_FUNCTIONS__
#include <stack-c.h>
#include <sciprint.h>
#include <MALLOC.h>
#include <Scierror.h>

#include "linear_model_scilab.h"

//#define DEBUG

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

void print_null(const char *s) {}
void print_string_scilab(const char *s) {sciprint(s);}

void exit_with_help()
{
	Scierror (999,
	"Usage: model = libsvm_lintrain(training_label_vector, training_instance_matrix, 'liblinear_options', 'col');\n"
	"Usage: model = libsvm_lintrain(weight_vector, training_label_vector, training_instance_matrix, 'liblinear_options', 'col');\n"
	"liblinear_options:\n"
	"-s type : set type of solver (default 1)\n"
	"	 0 -- L2-regularized logistic regression (primal)\n"
	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"	
	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
	"	 4 -- multi-class support vector classification by Crammer and Singer\n"
	"	 5 -- L1-regularized L2-loss support vector classification\n"
	"	 6 -- L1-regularized logistic regression\n"
	"	 7 -- L2-regularized logistic regression (dual)\n"
	"	11 -- L2-regularized L2-loss epsilon support vector regression (primal)\n"
	"	12 -- L2-regularized L2-loss epsilon support vector regression (dual)\n"
	"	13 -- L2-regularized L1-loss epsilon support vector regression (dual)\n"
	"-c cost : set the parameter C (default 1)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	"	-s 0 and 2\n" 
	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n" 
	"		where f is the primal function and pos/neg are # of\n" 
	"		positive/negative data (default 0.01)\n"
	"	-s 11\n"
	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n" 
	"	-s 1, 3, 4 and 7\n"
	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
	"	-s 5 and 6\n"
	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
	"		where f is the primal function (default 0.01)\n"
	"	-s 12 and 13\n"
	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
	"		where f is the dual function (default 0.1)\n"
	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	"col:\n"
	"	if 'col' is setted, training_instance_matrix is parsed in column format, otherwise is in row format\n"
	);
}

// liblinear arguments
struct parameter param_;		// set by parse_command_line
struct problem prob_;		// set by read_problem
struct model *model_;
struct feature_node *x_space_;
int cross_validation_flag;
int col_format_flag_;
int nr_fold_;
double bias;
int weight_vector_flag;

double do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob_.l);
	double retval = 0.0;

	cross_validation(&prob_,&param_,nr_fold_,target);
	if(param_.solver_type == L2R_L2LOSS_SVR || 
	   param_.solver_type == L2R_L1LOSS_SVR_DUAL || 
	   param_.solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		for(i=0;i<prob_.l;i++)
                {
                        double y = prob_.y[i];
                        double v = target[i];
                        total_error += (v-y)*(v-y);
                        sumv += v;
                        sumy += y;
                        sumvv += v*v;
                        sumyy += y*y;
                        sumvy += v*y;
                }
                #ifdef DEBUG
                sciprint("Cross Validation Mean squared error = %g\n",total_error/prob_.l);
                sciprint("Cross Validation Squared correlation coefficient = %g\n",
                        ((prob_.l*sumvy-sumv*sumy)*(prob_.l*sumvy-sumv*sumy))/
                        ((prob_.l*sumvv-sumv*sumv)*(prob_.l*sumyy-sumy*sumy))
                        );
		#endif
		retval = total_error/prob_.l;
	}
	else
	{
		for(i=0;i<prob_.l;i++)
			if(target[i] == prob_.y[i])
				++total_correct;
	  #ifdef DEBUG
	  sciprint("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob_.l);
          #endif
	  retval = 100.0*total_correct/prob_.l;
	}
	free(target);
	return retval;
}

// nrhs should be 3 or 4
int parse_command_line(int nrhs, const char *cmd, const char *cmd_col, char *model_file_name)
{
	int i, argc = 1;
	//char cmd[CMD_LEN];
	char *argv[CMD_LEN/2];
	void (*print_func)(const char *) = print_string_scilab;	// default printing to matlab display

	// default values
	param_.solver_type = L2R_L2LOSS_SVC_DUAL;
	param_.C = 1;
	param_.eps = INF; // see setting below
	param_.p = 0.1;
	param_.nr_weight = 0;
	param_.weight_label = NULL;
	param_.weight = NULL;
	cross_validation_flag = 0;
	col_format_flag_ = 0;
	bias = -1;


	if(nrhs <= 1 + weight_vector_flag)
		return 1;

	if(nrhs == 4 + weight_vector_flag)
	{
		//mxGetString(prhs[3], cmd, mxGetN(prhs[3])+1);
		if(strcmp(cmd_col, "col") == 0)
			col_format_flag_ = 1;
	}

	// put options in argv[]
	if(nrhs > 2 + weight_vector_flag)
	{
		//mxGetString(prhs[2], cmd,  mxGetN(prhs[2]) + 1);
		if((argv[argc] = strtok(cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
	}

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		if(i>=argc && argv[i-1][1] != 'q') // since option -q has no parameter
			return 1;
		switch(argv[i-1][1])
		{
			case 's':
				param_.solver_type = atoi(argv[i]);
				break;
			case 'c':
				param_.C = atof(argv[i]);
				break;
			case 'p':
				param_.p = atof(argv[i]);
				break;
			case 'e':
				param_.eps = atof(argv[i]);
				break;
			case 'B':
				bias = atof(argv[i]);
				break;
			case 'v':
				cross_validation_flag = 1;
				nr_fold_ = atoi(argv[i]);
				if(nr_fold_ < 2)
				{
					Scierror (999,"n-fold cross validation: n must >= 2\n");
					return 1;
				}
				break;
			case 'w':
				++param_.nr_weight;
				param_.weight_label = (int *) realloc(param_.weight_label,sizeof(int)*param_.nr_weight);
				param_.weight = (double *) realloc(param_.weight,sizeof(double)*param_.nr_weight);
				param_.weight_label[param_.nr_weight-1] = atoi(&argv[i-1][2]);
				param_.weight[param_.nr_weight-1] = atof(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			default:
				Scierror (999,"unknown option\n");
				return 1;
		}
	}

	set_print_string_function(print_func);

	if(param_.eps == INF)
	{
		switch(param_.solver_type)
		{
			case L2R_LR: 
			case L2R_L2LOSS_SVC:
				param_.eps = 0.01;
				break;
			case L2R_L2LOSS_SVR:
				param_.eps = 0.001;
				break;
			case L2R_L2LOSS_SVC_DUAL: 
			case L2R_L1LOSS_SVC_DUAL: 
			case MCSVM_CS: 
			case L2R_LR_DUAL: 
				param_.eps = 0.1;
				break;
			case L1R_L2LOSS_SVC: 
			case L1R_LR:
				param_.eps = 0.01;
				break;
			case L2R_L1LOSS_SVR_DUAL:
			case L2R_L2LOSS_SVR_DUAL:
				param_.eps = 0.1;
				break;
		}
	}
	return 0;
}



int read_problem_sparse(int *weight_vec, int *label_vec, int *instance_mat)
{
	int i, j, jj, k, low, high,r_labels, c_labels,r_samples, c_samples, r_weights,c_weights;
	int *ir, *jc;
	int elements, max_index, num_samples, label_vector_row_num, weight_vector_row_num;
	double *samples, *labels, *weights;
	int *instance_mat_col; // instance sparse matrix in column format
        SciErr _SciErr;

	prob_.x = NULL;
	prob_.y = NULL;
	prob_.W = NULL;
	x_space_ = NULL;



 	if(col_format_flag_)
	{
		Scierror (999,"training_instance_matrix in column format is not supported yet!\n");
		return -1;
	}
//		instance_mat_col = (int *)instance_mat;
// 	else
// 	{
// 		// transpose instance matrix
// 		mxArray *prhs[1], *plhs[1];
// 		prhs[0] = mxDuplicateArray(instance_mat);
// 		if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
// 		{
// 			mexPrintf("Error: cannot transpose training instance matrix\n");
// 			return -1;
// 		}
// 		instance_mat_col = plhs[0];
// 		mxDestroyArray(prhs[0]);
// 	}
        if (weight_vector_flag==1){
	  
	 _SciErr = getMatrixOfDouble(pvApiCtx, weight_vec, &r_weights, &c_weights, &weights);
	 if(_SciErr.iErr)
	{
			printError(&_SciErr, 0);
			return -1;
		}

	  
	} else {
	  r_weights=0;
	  c_weights=0;
	  weights=NULL;
	  weight_vector_row_num=0;
	}

        _SciErr = getMatrixOfDouble(pvApiCtx, label_vec, &r_labels, &c_labels, &labels);
	 if(_SciErr.iErr)
		{
			printError(&_SciErr, 0);
			return -1;
		}
	 _SciErr = getSparseMatrix(pvApiCtx,instance_mat,&r_samples, &c_samples, &num_samples, &ir, &jc, &samples);
	 if(_SciErr.iErr)
		{
			printError(&_SciErr, 0);
			return -1;
	}

		
		
	// the number of instance
	//prob.l = (int) mxGetN(instance_mat_col);
	prob_.l = r_samples;
	weight_vector_row_num = r_weights;
	label_vector_row_num = r_labels;//(int) mxGetM(label_vec);

	if(weight_vector_row_num == 0) 
		//sciprint("Warning: treat each instance with weight 1.0\n");
		;
	else if(weight_vector_row_num!=prob_.l)
	{
		sciprint("Length of weight vector does not match # of instances.\n");
		return -1;
	}
	
	if(label_vector_row_num!=prob_.l)
	{
		sciprint("Length of label vector does not match # of instances.\n");
		return -1;
	}
	
	// each column is one instance
	//labels = mxGetPr(label_vec);
	//samples = mxGetPr(instance_mat_col);
	//ir = mxGetIr(instance_mat_col);
	//jc = mxGetJc(instance_mat_col);

	//num_samples = (int) mxGetNzmax(instance_mat_col);

	elements = num_samples + prob_.l*2;
	max_index = c_samples;//(int) mxGetM(instance_mat_col);

	prob_.y = Malloc(double, prob_.l);
	prob_.W = Malloc(double,prob_.l);
	prob_.x = Malloc(struct feature_node*, prob_.l);
	x_space_ = Malloc(struct feature_node, elements);

	prob_.bias=bias;

	j = 0;jj=0;
	for(i=0;i<prob_.l;i++)
	{
		prob_.x[i] = &x_space_[j];
		prob_.y[i] = labels[i];
		prob_.W[i] = 1;
		if (weight_vector_row_num == prob_.l)
		  prob_.W[i] *= (double) weights[i];
		//low = (int) jc[i];
		low = 0;
		high = (int) ir[i];
		for(k=low;k<high;k++)
		{
			x_space_[j].index = (int) jc[jj];
			x_space_[j].value = samples[jj];
			j++;jj++;
	 	}
		if(prob_.bias>=0)
		{
			x_space_[j].index = max_index+1;
			x_space_[j].value = prob_.bias;
			j++;
		}
		x_space_[j++].index = -1;
	}

	if(prob_.bias>=0)
		prob_.n = max_index+1;
	else
		prob_.n = max_index;

	return 0;
}

// Interface function of matlab
// now assume prhs[0]: label prhs[1]: features
//void mexFunction( int nlhs, mxArray *plhs[],
//		int nrhs, const mxArray *prhs[] )
int sci_train(char * fname)
{
        SciErr _SciErr;
	const char *error_msg;
	int * p_weight_vector = NULL;
	int * p_label_vector = NULL;
        int * p_instance_matrix = NULL;
	int * p_option_string = NULL;
	int * p_col_string = NULL;
	int r_samples, c_samples;
	double *samples = NULL;
        int type,type3;
	char * option_string = NULL;
	char * col_string = NULL;
	int rhs_offset=0;
	// fix random seed to have same results for each run
	// (for cross validation)
	srand(1);
	// check for weight vector
	weight_vector_flag=0;
	if (Rhs>2) {
	  _SciErr = getVarAddressFromPosition(pvApiCtx, 3, &p_option_string);
		if(_SciErr.iErr)
		{
			printError(&_SciErr, 0);
			return 0;
		}
                _SciErr = getVarType(pvApiCtx, p_option_string, &type);
		if(_SciErr.iErr)
		{
			printError(&_SciErr, 0);
			return 0;
		}
	        if (type!=sci_strings){
		  weight_vector_flag=1;
		  rhs_offset=1;
		  p_instance_matrix=p_option_string;
		  p_option_string=NULL;
		} else
		  getAllocatedSingleString(pvApiCtx, p_option_string, &option_string);
	  
	}

	// Transform the input Matrix to libsvm format
	if(Rhs > (1+rhs_offset) && (Rhs < 5+rhs_offset))
	{
		int err=0;
		if ( weight_vector_flag==1){
		  		_SciErr = getVarAddressFromPosition(pvApiCtx, 1, &p_weight_vector);
			      if(_SciErr.iErr)
			      {
				      printError(&_SciErr, 0);
				      return 0;
			      }
			      _SciErr = getVarType(pvApiCtx, p_weight_vector, &type);
			      if(_SciErr.iErr)
			      {
				      printError(&_SciErr, 0);
				      return 0;
			      }
			      if (type!=sci_matrix)
			      {
				Scierror (999,"Error: weight vector must be double\n");	
				return 0;
			      }
			      
		  
		}

		_SciErr = getVarAddressFromPosition(pvApiCtx, 1+rhs_offset, &p_label_vector);
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
		if (type!=sci_matrix)
		{
		  Scierror (999,"Error: label vector must be double\n");	
		  return 0;
		}
		if (weight_vector_flag==0) {
		  _SciErr = getVarAddressFromPosition(pvApiCtx, 2+rhs_offset, &p_instance_matrix);
		  if(_SciErr.iErr)
		  {
			  printError(&_SciErr, 0);
			  return 0;
		  }
		}
                _SciErr = getVarType(pvApiCtx, p_instance_matrix, &type);
		if(_SciErr.iErr)
		{
			printError(&_SciErr, 0);
			return 0;
		}
		
		 if (type!=sci_matrix && type!=sci_sparse)
		{
		 Scierror (999,"Error: instance matrix must be double or sparse\n");			
		  return 0;
		}

		
		if (Rhs==(4+rhs_offset)) {
		    _SciErr = getVarAddressFromPosition(pvApiCtx, 4+rhs_offset, &p_col_string);
		    if(_SciErr.iErr)
		    {
			    printError(&_SciErr, 0);
			    return 0;
		    }
		    _SciErr = getVarType(pvApiCtx, p_col_string, &type3);
		    if(_SciErr.iErr)
		    {
			    printError(&_SciErr, 0);
			    return 0;
		    }
		 if (type3==sci_strings)
		  {
		    getAllocatedSingleString(pvApiCtx, p_col_string, &col_string);
		}

		    
		}
		if (weight_vector_flag==1) {
		if (Rhs>(2+rhs_offset)) {
		    _SciErr = getVarAddressFromPosition(pvApiCtx, 3+rhs_offset, &p_option_string);
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
		}
		
		if(parse_command_line(Rhs,option_string,col_string, NULL))
		{

			destroy_param(&param_);
			exit_with_help();
			return 0;
		}
		if (option_string!=NULL)
		 freeAllocatedSingleString(option_string);
		if (col_string!=NULL)
                 freeAllocatedSingleString(col_string);

		if(type==sci_sparse)
			err = read_problem_sparse(p_weight_vector,p_label_vector, p_instance_matrix);
		else
		{
			destroy_param(&param_);	
			Scierror (999,"Training_instance_matrix must be sparse\n");
			return 0;
		}

		// train's original code
		error_msg = check_parameter(&prob_, &param_);

		if(err || error_msg)
		{
			if (error_msg != NULL)
				Scierror (999,"Error: %s\n", error_msg);
			else
			        Scierror (999,"Error\n");
			destroy_param(&param_);
			free(prob_.y);
			free(prob_.x);
			free(x_space_);
			return 0;
		}

		if(cross_validation_flag)
		{
			double *ptr;
			_SciErr = allocMatrixOfDouble(pvApiCtx, Rhs + 1, 1, 1, &ptr);
			if(_SciErr.iErr)
		        {
			    printError(&_SciErr, 0);
			    return 0;
		         }
			//plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
			//ptr = mxGetPr(plhs[0]);
			ptr[0] = do_cross_validation();
			LhsVar(1) = Rhs + 1;
			  /* This function put on scilab stack, the lhs variable
			which are at the position lhs(i) on calling stack */
			/* You need to add PutLhsVar here because WITHOUT_ADD_PUTLHSVAR 
			was defined and equal to %t */
			/* without this, you do not need to add PutLhsVar here */
			PutLhsVar();
		}
		else
		{
			//const char *error_msg;

			model_ = train(&prob_, &param_);
			_SciErr = linear_model_to_scilab_structure( model_);
			if(_SciErr.iErr){
				//Scierror (999,"Error: can't convert libsvm model to matrix structure: %s\n", error_msg);
			        printError(&_SciErr, 0);
				exit_with_help();

			}else{
			    /* This function put on scilab stack, the lhs variable
		    which are at the position lhs(i) on calling stack */
		    /* You need to add PutLhsVar here because WITHOUT_ADD_PUTLHSVAR 
		    was defined and equal to %t */
		    /* without this, you do not need to add PutLhsVar here */
		    	LhsVar(1) = Rhs+1; 
			PutLhsVar();
			}
			free_and_destroy_model(&model_);
		}
		destroy_param(&param_);
		free(prob_.y);
		free(prob_.x);
		free(prob_.W);
		free(x_space_);
		return 0;
	}
	else
	{
		exit_with_help();
		return 0;
	}
}
