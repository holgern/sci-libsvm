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
#include "linear.h"


#include <api_scilab.h>
#include <stack-c.h>
#include <sciprint.h>
#include <MALLOC.h>
#include <Scierror.h>

#include "linear_model_scilab.h"


#define CMD_LEN 2048

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

int col_format_flag;

// void read_sparse_instance(const mxArray *prhs, int index, struct feature_node *x, int feature_number, double bias)
// {
// 	int i, j, low, high;
// 	mwIndex *ir, *jc;
// 	double *samples;
// 
// 	ir = mxGetIr(prhs);
// 	jc = mxGetJc(prhs);
// 	samples = mxGetPr(prhs);
// 
// 	// each column is one instance
// 	j = 0;
// 	low = (int) jc[index], high = (int) jc[index+1];
// 	for(i=low; i<high && (int) (ir[i])<feature_number; i++)
// 	{
// 		x[j].index = (int) ir[i]+1;
// 		x[j].value = samples[i];
// 		j++;
// 	}
// 	if(bias>=0)
// 	{
// 		x[j].index = feature_number+1;
// 		x[j].value = bias;
// 		j++;
// 	}
// 	x[j].index = -1;
// }

static void fake_answer()
{
	LhsVar(1) = 0;
	LhsVar(2) = 0;
	LhsVar(3) = 0;
}

void do_predict(int *label_vec,  int *instance_mat, struct model *model_, const int predict_probability_flag)
{
	int label_vector_row_num, label_vector_col_num;
	int feature_number, testing_instance_number;
	int instance_index;
	int r_labels, c_labels,r_samples, c_samples;
	 int type;
	double *ptr_instance, *ptr_label, *ptr_predict_label;
	double *ptr_prob_estimates, *ptr_dec_values, *ptr;
	struct feature_node *x;
	 SciErr _SciErr;
	   int *ir, *jc;
	   int jj,num_samples;
	//mxArray *pplhs[1]; // instance sparse matrix in row format

	int correct = 0;
	int total = 0;

	int nr_class=get_nr_class(model_);
	int nr_w;
	double *prob_estimates=NULL;

	if(nr_class==2 && model_->param.solver_type!=MCSVM_CS)
		nr_w=1;
	else
		nr_w=nr_class;

	_SciErr = getMatrixOfDouble(pvApiCtx, label_vec, &r_labels, &c_labels, &ptr_label);
	 //_SciErr = getSparseMatrix(pvApiCtx,instance_mat,&r_samples, &c_samples, &num_samples, &ir, &jc, &ptr_instance);
        _SciErr = getVarType(pvApiCtx, instance_mat, &type);
	 if (type==sci_sparse)
	   _SciErr = getSparseMatrix(pvApiCtx,instance_mat,&r_samples, &c_samples, &num_samples, &ir, &jc, &ptr_instance);
         else
	   sciprint("Testing_instance_matrix must be sparse\n");
	
	
	
	// prhs[1] = testing instance matrix
	feature_number = get_nr_feature(model_);
	testing_instance_number = r_samples;//(int) mxGetM(prhs[1]);
	if(col_format_flag)
	{
		feature_number = r_samples;//(int) mxGetM(prhs[1]);
		testing_instance_number = c_samples;//(int) mxGetN(prhs[1]);
	}

	label_vector_row_num = r_labels;//(int) mxGetM(prhs[0]);
	label_vector_col_num = c_labels;//(int) mxGetN(prhs[0]);

	if(label_vector_row_num!=testing_instance_number)
	{
		sciprint("Length of label vector does not match # of instances.\n");
		fake_answer();
		return;
	}
	if(label_vector_col_num!=1)
	{
		sciprint("label (1st argument) should be a vector (# of column is 1).\n");
		fake_answer();
		return;
	}

	//ptr_instance = mxGetPr(prhs[1]);
	//ptr_label    = mxGetPr(prhs[0]);

	// transpose instance matrix
// 	if(mxIsSparse(prhs[1]))
// 	{
// 		if(col_format_flag)
// 		{
// 			pplhs[0] = (mxArray *)prhs[1];
// 		}
// 		else
// 		{
// 			mxArray *pprhs[1];
// 			pprhs[0] = mxDuplicateArray(prhs[1]);
// 			if(mexCallMATLAB(1, pplhs, 1, pprhs, "transpose"))
// 			{
// 				mexPrintf("Error: cannot transpose testing instance matrix\n");
// 				fake_answer(plhs);
// 				return;
// 			}
// 		}
// 	}
// 	else
// 		mexPrintf("Testing_instance_matrix must be sparse\n");


	prob_estimates = Malloc(double, nr_class);

	//plhs[0] = mxCreateDoubleMatrix(testing_instance_number, 1, mxREAL);
	ptr_predict_label = (double*)malloc(sizeof(double) * testing_instance_number*1);
	if(predict_probability_flag)
		//plhs[2] = mxCreateDoubleMatrix(testing_instance_number, nr_class, mxREAL);
		ptr_prob_estimates =  (double*)malloc(sizeof(double) * testing_instance_number*nr_class);
	else
		//plhs[2] = mxCreateDoubleMatrix(testing_instance_number, nr_w, mxREAL);
	        ptr_dec_values =  (double*)malloc(sizeof(double) * testing_instance_number*nr_w);

	//ptr_predict_label = mxGetPr(plhs[0]);
	//ptr_prob_estimates = mxGetPr(plhs[2]);
	//ptr_dec_values = mxGetPr(plhs[2]);
	x = Malloc(struct feature_node, feature_number+2);
	jj=0;
	for(instance_index=0;instance_index<testing_instance_number;instance_index++)
	{
		int i,j;
		double target,v;
                int low,high;
		
		target = ptr_label[instance_index];

		// prhs[1] and prhs[1]^T are sparse
		//read_sparse_instance(pplhs[0], instance_index, x, feature_number, model_->bias);
		  
		    j = 0;
		    low = 0;
		    high = ir[instance_index];
		    for(i=low;i<high &&(int) (jc[jj])<(feature_number + 1);i++)
		    {
			    x[j].index = jc[jj];
			    x[j].value = ptr_instance[jj];
			    j++;jj++;
		    }
		    if(model_->bias>=0)
		    {
			    x[j].index = feature_number+1;
			    x[j].value = model_->bias;
			    j++;
		    }
		    x[j++].index = -1;
		
		
		

		if(predict_probability_flag)
		{
			v = predict_probability(model_, x, prob_estimates);
			ptr_predict_label[instance_index] = v;
			for(i=0;i<nr_class;i++)
				ptr_prob_estimates[instance_index + i * testing_instance_number] = prob_estimates[i];
		}
		else
		{
			double *dec_values = Malloc(double, nr_class);
			v = predict(model_, x);
			ptr_predict_label[instance_index] = v;

			predict_values(model_, x, dec_values);
			for(i=0;i<nr_w;i++)
				ptr_dec_values[instance_index + i * testing_instance_number] = dec_values[i];
			free(dec_values);
		}

		if(v == target)
			++correct;
		++total;
	}
	sciprint("Accuracy = %g%% (%d/%d)\n", (double) correct/total*100,correct,total);

	// return accuracy, mean squared error, squared correlation coefficient
	//plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	//ptr = mxGetPr(plhs[1]);
	ptr = (double*)malloc(sizeof(double) * 1*1);
	ptr[0] = (double) correct/total*100;

	createMatrixOfDouble(pvApiCtx, Rhs + 1, testing_instance_number, 1, ptr_predict_label);
        free(ptr_predict_label);
        LhsVar(1) = Rhs + 1; 
	
	if (Lhs	> 1){
	    createMatrixOfDouble(pvApiCtx, Rhs + 2, 1, 1, ptr);
	    free(ptr);
	    LhsVar(2) = Rhs + 2; 
        } else
           free(ptr);
	
	if (Lhs > 2) {
	    if(predict_probability_flag){
	      createMatrixOfDouble(pvApiCtx, Rhs + 3, testing_instance_number, nr_class, ptr_prob_estimates);
	      free(ptr_prob_estimates);
	    }else{
	      createMatrixOfDouble(pvApiCtx, Rhs + 3, testing_instance_number, nr_w, ptr_dec_values);
	      free(ptr_dec_values);
	    }
	    LhsVar(3) = Rhs + 3; 
	} else
	  if(predict_probability_flag)
	    free(ptr_prob_estimates);
	  else
	    free(ptr_dec_values);
	
	
	 /* This function put on scilab stack, the lhs variable
			which are at the position lhs(i) on calling stack */
			/* You need to add PutLhsVar here because WITHOUT_ADD_PUTLHSVAR 
			was defined and equal to %t */
			/* without this, you do not need to add PutLhsVar here */
			PutLhsVar();
	free(x);
	if(prob_estimates != NULL)
		free(prob_estimates);
}

void lin_exit_with_help()
{
	sciprint(
			"Usage: [predicted_label, accuracy, decision_values/prob_estimates] = predict(testing_label_vector, testing_instance_matrix, model, 'liblinear_options','col')\n"
			"liblinear_options:\n"
			"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0)\n"
			"col:\n"
			"	if 'col' is setted testing_instance_matrix is parsed in column format, otherwise is in row format"
			);
}

int  sci_predict(char * fname)
{
	int prob_estimate_flag = 0;
	struct model *model_;
	char *cmd;
	char * col_cmd;
	
	
	int * p_label_vector = NULL;
        int * p_instance_matrix = NULL;
	int * p_model = NULL;
	int * p_option_string = NULL;
	int * p_col_string = NULL;
	int type,type3;
	SciErr _SciErr;
	
	col_format_flag = 0;
	
        
	col_format_flag = 0;
	if(Rhs > 5 || Rhs < 3)
	{
		lin_exit_with_help();
		fake_answer();
		return 0;
	}
	if(Rhs == 5)
	{
		//mxGetString(prhs[4], cmd, mxGetN(prhs[4])+1);
		_SciErr = getVarAddressFromPosition(pvApiCtx, 5, &p_col_string);
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
		    getAllocatedSingleString(pvApiCtx, p_col_string, &col_cmd);
		    if(strcmp(col_cmd, "col") == 0)
		    {			
			    col_format_flag = 1;
		     }
		    if (col_cmd!=NULL)
		     freeAllocatedSingleString(col_cmd);
		}
	}
	
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
		  sciprint("Error: label vector must be double\n");	
		   fake_answer();
		  return 0;
		}
		_SciErr = getVarAddressFromPosition(pvApiCtx, 2, &p_instance_matrix);
		  if(_SciErr.iErr)
		      {
			printError(&_SciErr, 0);
			return 0;
		      } 
                _SciErr = getVarType(pvApiCtx, p_instance_matrix, &type);
		 if (type!=sci_sparse)
		{
		  sciprint("Error: Testing_instance_matrix  must be sparse\n");			
		   fake_answer();
		  return 0;
		}
		
                 _SciErr = getVarAddressFromPosition(pvApiCtx, 3, &p_model);
		  if(_SciErr.iErr)
		      {
			printError(&_SciErr, 0);
			return 0;
		      } 
                _SciErr = getVarType(pvApiCtx, p_model, &type);
		  if(_SciErr.iErr)
		      {
			printError(&_SciErr, 0);
			return 0;
		      } 
       


	if(type == sci_mlist | type == sci_list )
	{
		const char *error_msg;

		// parse options
		if(Rhs>=4)
		{
			int i, argc = 1;
			char *argv[CMD_LEN/2];

			// put options in argv[]
			//mxGetString(prhs[3], cmd,  mxGetN(prhs[3]) + 1);
			_SciErr = getVarAddressFromPosition(pvApiCtx, 4, &p_option_string);
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
		    getAllocatedSingleString(pvApiCtx, p_option_string, &cmd);
		    
		    
			if((argv[argc] = strtok(cmd, " ")) != NULL)
				while((argv[++argc] = strtok(NULL, " ")) != NULL)
					;

			for(i=1;i<argc;i++)
			{
				if(argv[i][0] != '-') break;
				if(++i>=argc)
				{
					lin_exit_with_help();
					fake_answer();
					return;
				}
				switch(argv[i-1][1])
				{
					case 'b':
						prob_estimate_flag = atoi(argv[i]);
						break;
					default:
						sciprint("unknown option\n");
						lin_exit_with_help();
						fake_answer();
						return;
				}
			}
			freeAllocatedSingleString(cmd);
		}
		}

		model_ = Malloc(struct model, 1);
		error_msg = scilab_matrix_to_linear_model(p_model,model_);
		if(error_msg)
		{
			sciprint("Error: can't read model: %s\n", error_msg);
			free_and_destroy_model(&model_);
			fake_answer();
			return;
		}

		if(prob_estimate_flag)
		{
			if(!check_probability_model(model_))
			{
				sciprint("probability output is only supported for logistic regression\n");
				prob_estimate_flag=0;
			}
		}

// 		if(mxIsSparse(prhs[1]))
			do_predict(p_label_vector, p_instance_matrix, model_, prob_estimate_flag);
// 		else
// 		{
// 			sciprint("Testing_instance_matrix must be sparse\n");
// 			fake_answer();
// 		}

		// destroy model_
		free_and_destroy_model(&model_);
	}
	else
	{
		sciprint("model file should be a struct array\n");
		fake_answer();
	}

	return 0;
}
