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
#include "svm.h"

#include <api_scilab.h>
// #define __USE_DEPRECATED_STACK_FUNCTIONS__
// #include <stack-c.h>
#include <sciprint.h>
#include <MALLOC.h>
#include <Scierror.h>


#include "svm_model_scilab.h"

#define CMD_LEN 2048


//#define DEBUG

// void read_sparse_instance(int *instance_mat, int index, struct svm_node *x)
// {
//         int i, j, low, high;
// 	int r_samples, c_samples,num_samples;
//         int *ir, *jc;
//         double *samples;
// 
//         //ir = mxGetIr(prhs);
// 	//jc = mxGetJc(prhs);
// 	//samples = mxGetPr(prhs);
// 	 _SciErr = getSparseMatrix(pvApiCtx,instance_mat,&r_samples, &c_samples, &num_samples, &ir, &jc, &samples);
// 
//         // each column is one instance
//         j = 0;
//         low = 0, high = ir[index];
//         for(i=low;i<high;i++)
//         {
//                 x[j].index = jc[i];
//                 x[j].value = samples[i];
//                 j++;
//         }
// 	x[j].index = -1;
// }


void do_predict_svm(int *label_vec,  int *instance_mat, struct svm_model *model, const int predict_probability)
{
	int label_vector_row_num, label_vector_col_num;
	int feature_number, testing_instance_number;
	int instance_index;
	int r_labels, c_labels,r_samples, c_samples;
	 int type;
	double *ptr_instance, *ptr_label, *ptr_predict_label; 
	double *ptr_prob_estimates, *ptr_dec_values, *ptr;
	struct svm_node *x;
	 SciErr _SciErr;
//	mxArray *pplhs[1]; // transposed instance sparse matrix

	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
        double *prob_estimates=NULL;
	int m_out3,n_out3;
	 int k, j, low, high,jj,num_samples;
	  int *ir, *jc;

        // prhs[1] = testing instance matrix
	
	 _SciErr = getMatrixOfDouble(pvApiCtx, label_vec, &r_labels, &c_labels, &ptr_label);
	 //_SciErr = getSparseMatrix(pvApiCtx,instance_mat,&r_samples, &c_samples, &num_samples, &ir, &jc, &ptr_instance);
         _SciErr = getVarType(pvApiCtx, instance_mat, &type);
	 
	 if (type==sci_sparse)
	   _SciErr = getSparseMatrix(pvApiCtx,instance_mat,&r_samples, &c_samples, &num_samples, &ir, &jc, &ptr_instance);
         else
	   _SciErr = getMatrixOfDouble(pvApiCtx, instance_mat, &r_samples, &c_samples, &ptr_instance);
	 
        feature_number = c_samples;// mxGetN(prhs[1]);
        testing_instance_number = r_samples;//mxGetM(prhs[1]);
        label_vector_row_num = r_labels;//mxGetM(prhs[0]);
        label_vector_col_num = c_labels;//mxGetN(prhs[0]);

        if(label_vector_row_num!=testing_instance_number)
        {
		Scierror (999,"Length of label vector does not match # of instances.\n");
		return;
	}
	if(label_vector_col_num!=1)
	{
		Scierror (999,"label (1st argument) should be a vector (# of column is 1).\n");
		return;
	}

	//ptr_instance = mxGetPr(prhs[1]);
	//ptr_label    = mxGetPr(prhs[0]);
	

	// transpose instance matrix
	if(type==sci_sparse)
	{
		if(model->param.kernel_type == PRECOMPUTED)
		{
		        Scierror (999,"Error: Precomputed kernel requires dense matrix\n");	
		            return;
			    /*
                        // precomputed kernel requires dense matrix, so we make one
                        mxArray *rhs[1], *lhs[1];
                        rhs[0] = mxDuplicateArray(prhs[1]);
                        if(mexCallSCILAB(1, lhs, 1, rhs, "full"))
                        {
                                sciprint("Error: cannot full testing instance matrix\n");
                                fake_answer();
				return;
			}
			ptr_instance = mxGetPr(lhs[0]);
			mxDestroyArray(rhs[0]);
			*/
		}
		else
                {
//                         mxArray *pprhs[1];
//                         pprhs[0] = mxDuplicateArray(prhs[1]);
//                         if(mexCallSCILAB(1, pplhs, 1, pprhs, "transpose"))
//                         {
//                                 sciprint("Error: cannot transpose testing instance matrix\n");
//                                 fake_answer();
// 				return;
// 			}
		}
	}

	if(predict_probability)
	{
		if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
			sciprint("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
	}

	//plhs[0] = mxCreateDoubleMatrix(testing_instance_number, 1, mxREAL);
	 	  
	  
	if(predict_probability)
	{
		// prob estimates are in plhs[2]
		if(svm_type==C_SVC || svm_type==NU_SVC){
			//plhs[2] = mxCreateDoubleMatrix(testing_instance_number, nr_class, mxREAL);
		        ptr_prob_estimates =  (double*)malloc(sizeof(double) * testing_instance_number*nr_class);
			m_out3=testing_instance_number;
			n_out3=nr_class;
		}else{
			//plhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL);
		        ptr_prob_estimates =  NULL;
			m_out3=0;
			n_out3=0;
		}
	}
	else
	{
		// decision values are in plhs[2]
		if(svm_type == ONE_CLASS ||
		   svm_type == EPSILON_SVR ||
		   svm_type == NU_SVR ||
		   nr_class == 1) // if only one class in training data, decision values are still returned.
		{
			//plhs[2] = mxCreateDoubleMatrix(testing_instance_number, 1, mxREAL);
			ptr_dec_values =  (double*)malloc(sizeof(double) * testing_instance_number*1);
			m_out3=testing_instance_number;
			n_out3=1;
		}else{
			//plhs[2] = mxCreateDoubleMatrix(testing_instance_number, nr_class*(nr_class-1)/2, mxREAL);
		        ptr_dec_values =  (double*)malloc(sizeof(double) * testing_instance_number*(nr_class*(nr_class-1)/2));
			m_out3=testing_instance_number;
			n_out3=(nr_class*(nr_class-1)/2);
		}
	}

	ptr_predict_label = (double*)malloc(sizeof(double) * testing_instance_number*1);
	//ptr_prob_estimates = mxGetPr(plhs[2]);
	//ptr_dec_values = mxGetPr(plhs[2]);
	x = (struct svm_node*)malloc((feature_number+1)*sizeof(struct svm_node) );
	jj=0;
	for(instance_index=0;instance_index<testing_instance_number;instance_index++)
	{
		int i;
		double target_label, predict_label;

		target_label = ptr_label[instance_index];

		if(type==sci_sparse && model->param.kernel_type != PRECOMPUTED){ // prhs[1]^T is still sparse
			//read_sparse_instance(instance_mat, instance_index, x);
		
		  
		   
		    //int r_samples, c_samples,num_samples;
		    //int *ir, *jc;
		    //double *samples;

		    //ir = mxGetIr(prhs);
		    //jc = mxGetJc(prhs);
		    //samples = mxGetPr(prhs);
		    //_SciErr = getSparseMatrix(pvApiCtx,instance_mat,&r_samples, &c_samples, &num_samples, &ir, &jc, &samples);

		    // each column is one instance
		    j = 0;
		    low = 0, high = ir[instance_index];
		    for(i=low;i<high;i++)
		    {
			    x[j].index = jc[jj];
			    x[j].value = ptr_instance[jj];
			    j++;jj++;
		    }
		    x[j++].index = -1;
			
			
		}
		else
		{
			for(i=0;i<feature_number;i++)
			{
				x[i].index = i+1;
				x[i].value = ptr_instance[testing_instance_number*i+instance_index];
			}
			x[feature_number].index = -1;
		}

		if(predict_probability)
		{
			if(svm_type==C_SVC || svm_type==NU_SVC)
			{
				predict_label = svm_predict_probability(model, x, prob_estimates);
				ptr_predict_label[instance_index] = predict_label;
				for(i=0;i<nr_class;i++)
					ptr_prob_estimates[instance_index + i * testing_instance_number] = prob_estimates[i];
			} else {
				predict_label = svm_predict(model,x);
				ptr_predict_label[instance_index] = predict_label;
			}
                }
                else
                {
                        if(svm_type == ONE_CLASS ||
                           svm_type == EPSILON_SVR ||
                           svm_type == NU_SVR)
                        {
                                double res;
                                predict_label = svm_predict_values(model, x, &res);
                                ptr_dec_values[instance_index] = res;
                        }
                        else
                        {
                                double *dec_values = (double *) malloc(sizeof(double) * nr_class*(nr_class-1)/2);
                                predict_label = svm_predict_values(model, x, dec_values);
			
				if(nr_class == 1) 
					ptr_dec_values[instance_index] = 1;
				else
					for(i=0;i<(nr_class*(nr_class-1))/2;i++)
						ptr_dec_values[instance_index + i * testing_instance_number] = dec_values[i];
                                free(dec_values);
                        }
                        ptr_predict_label[instance_index] = predict_label;
                }

                if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	#ifdef DEBUG
	if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		sciprint("Mean squared error = %g (regression)\n",error/total);
		sciprint("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else
		sciprint("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct/total*100,correct,total);
        #endif
	// return accuracy, mean squared error, squared correlation coefficient
	//plhs[1] = mxCreateDoubleMatrix(3, 1, mxREAL);
	ptr = (double*)malloc(sizeof(double) * 3*1);
	ptr[0] = (double)correct/total*100;
	ptr[1] = error/total;
	ptr[2] = ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
				((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt));

	createMatrixOfDouble(pvApiCtx, nbInputArgument(pvApiCtx) + 1, testing_instance_number, 1, ptr_predict_label);
        free(ptr_predict_label);
        AssignOutputVariable(pvApiCtx,1) = nbInputArgument(pvApiCtx) + 1; 
	
if (nbOutputArgument(pvApiCtx)	> 1){
	createMatrixOfDouble(pvApiCtx, nbInputArgument(pvApiCtx) + 2, 3, 1, ptr);
	free(ptr);
	AssignOutputVariable(pvApiCtx,2) = nbInputArgument(pvApiCtx) + 2; 
} else
  free(ptr);

if (nbOutputArgument(pvApiCtx) > 2) {
	if(predict_probability){
	  createMatrixOfDouble(pvApiCtx, nbInputArgument(pvApiCtx) + 3, m_out3, n_out3, ptr_prob_estimates);
	   free(ptr_prob_estimates);
	}else{
	   createMatrixOfDouble(pvApiCtx, nbInputArgument(pvApiCtx) + 3, m_out3, n_out3, ptr_dec_values);
	   free(ptr_dec_values);
	}
	AssignOutputVariable(pvApiCtx,3) = nbInputArgument(pvApiCtx) + 3; 
} else
  if(predict_probability)
     free(ptr_prob_estimates);
  else
    free(ptr_dec_values);
  
	  /* This function put on scilab stack, the lhs variable
  which are at the position lhs(i) on calling stack */
  /* You need to add PutLhsVar here because WITHOUT_ADD_PUTLHSVAR 
  was defined and equal to %t */
  /* without this, you do not need to add PutLhsVar here */
  ReturnArguments(pvApiCtx);
  
	free(x);
	if(prob_estimates != NULL)
                free(prob_estimates);
}

void exit_with_help_predict()
{
       Scierror (999,
                "Usage: [predicted_label, accuracy, decision_values/prob_estimates] = libsvm_svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')\n"
		"Parameters:\n"
		"  model: SVM model structure from svmtrain.\n"
		"  libsvm_options:\n"
		"    -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet\n"
		"Returns:\n"
		"  predicted_label: SVM prediction output vector.\n"
		"  accuracy: a vector with accuracy, mean squared error, squared correlation coefficient.\n"
		"  prob_estimates: If selected, probability estimate vector.\n"
        );
}


int sci_svmpredict(char * fname)


{ 
        SciErr _SciErr;
        int prob_estimate_flag = 0;
        struct svm_model *model;
	int * p_label_vector = NULL;
        int * p_instance_matrix = NULL;
	int * p_model = NULL;
	int * p_option_string = NULL;
	int type,type3;
	
        char *cmd  = NULL;

        if(nbInputArgument(pvApiCtx) > 4 || nbInputArgument(pvApiCtx) < 3)
        {
                exit_with_help_predict();
                return 0;
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
		  Scierror (999,"Error: label vector must be double\n");	
		  return 0;
		}
		_SciErr = getVarAddressFromPosition(pvApiCtx, 2, &p_instance_matrix);
		  if(_SciErr.iErr)
		      {
			printError(&_SciErr, 0);
			return 0;
		      } 
                _SciErr = getVarType(pvApiCtx, p_instance_matrix, &type);
		 if (type!=sci_matrix && type!=sci_sparse)
		{
		  Scierror (999,"Error: instance matrix must be double\n");			
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
		if(nbInputArgument(pvApiCtx)==4)
		{
			int i, argc = 1;
			//char cmd[CMD_LEN], *argv[CMD_LEN/2];
			char  *argv[CMD_LEN/2];

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
                                        exit_with_help_predict();
                                        return 0;
                                }
				switch(argv[i-1][1])
				{
					case 'b':
						prob_estimate_flag = atoi(argv[i]);
                                                break;
                                        default:
                                                sciprint ("Unknown option: -%c\n", argv[i-1][1]);
                                                exit_with_help_predict();
                                                return 0;
                                }
                        }
                       if (cmd!=NULL)
                        freeAllocatedSingleString(cmd);
		   }
                }
                
		#ifdef DEBUG
                    printf("DEBUG: start\n");
                 #endif 
		    
                model = scilab_matrix_to_model(p_model, &error_msg);
                if (model == NULL)
                {
                        Scierror (999,"Error: can't read model: %s\n", error_msg);
			return 0;
		}
		#ifdef DEBUG
                    printf("DEBUG: read model done\n");
                 #endif 

		if(prob_estimate_flag)
		{
			if(svm_check_probability_model(model)==0)
                        {
                                Scierror(999,"Model does not support probabiliy estimates\n");
                                svm_free_and_destroy_model(&model);
                                return 0;
                        }
                }
		else
		{
			if(svm_check_probability_model(model)!=0)
				printf("Model supports probability estimates, but disabled in predicton.\n");
		}
		
		#ifdef DEBUG
                    printf("DEBUG: check probability done\n");
                 #endif 

                do_predict_svm(p_label_vector,p_instance_matrix,model, prob_estimate_flag);
		#ifdef DEBUG
                    printf("DEBUG: predict done\n");
                 #endif 
                // destroy model
                svm_free_and_destroy_model(&model);
        }
        else
        {
		Scierror (999,"model file should be a struct array\n");
	}

	return 0;
}
