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

#include <stdlib.h>
#include <string.h>
#include "svm.h"

#include <api_scilab.h>
// #include <stack-c.h>
#include <sciprint.h>
#include <MALLOC.h>
#include <Scierror.h>

#define NUM_OF_RETURN_FIELD 10

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#define USE_SPARSE_IN_STRUCT 0

static const char *field_names[] = {
        "st",
	"dims",
	"Parameters",
	"nr_class",
	"totalSV",
	"rho",
	"Label",
	"ProbA",
	"ProbB",
	"nSV",
	"sv_coef",
        "SVs"
}; 

typedef struct scisparse {
         int m;
         int n;
         int it;
         int nel; 
         int *mnel;
         int *icol; 
         double *R; 
         double *I ; 
 } SciSparse ;

SciErr model_to_scilab_structure( int num_of_feature, struct svm_model *model)
{
        int i, j, n;
        double *ptr;
	double *return_model, **rhs;
	int out_id = 0;
	int *piAddr             = NULL;
	SciErr _SciErr;
	int jc_index,ir_index, nonzero_element;
        //int *ir, *jc;
	double  *dims;
	
        static SciSparse * ConstrMat = NULL;

	
	rhs = (double **)MALLOC(sizeof(double *)*NUM_OF_RETURN_FIELD);
	
	
	
	_SciErr = createMList(pvApiCtx,Rhs + 1,12, &piAddr);
      if(_SciErr.iErr)
	      {
		      //printError(&_SciErr, 0);
		      return _SciErr;
	      }

	  _SciErr = createMatrixOfStringInList(pvApiCtx, Rhs + 1, piAddr, 1, NUM_OF_RETURN_FIELD+2, 1, field_names);
           if(_SciErr.iErr)
	      {
		      //printError(&_SciErr, 0);
		      return _SciErr;
	      }
	  
	  //dims
	      //dims = (double *)MALLOC(sizeof(double)*2);
	      // dims[0]=1;
	      //dims[1]=1;
	   _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 2, 1, 2 , &dims);
           if(_SciErr.iErr)
	      {
		      //printError(&_SciErr, 0);
		      return _SciErr;
	      }
	       dims[0]=1;
	       dims[1]=1;
	
       
	
	  // Parameters

	 _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 3, 5, 1, &rhs[out_id]);
           if(_SciErr.iErr)
	      {
		      //printError(&_SciErr, 0);
		      return _SciErr;
	      }
	
	// Parameters
	//rhs[out_id] = mxCreateDoubleMatrix(5, 1, mxREAL);
	//rhs[out_id] = (double *)MALLOC(5*1*sizeof(double));
	ptr = (rhs[out_id]);
	ptr[0] = model->param.svm_type;
	ptr[1] = model->param.kernel_type;
	ptr[2] = model->param.degree;
	ptr[3] = model->param.gamma;
	ptr[4] = model->param.coef0;
	

	
	 // nr_class
        out_id++;
	 _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 4, 1, 1, &rhs[out_id]);
           if(_SciErr.iErr)
	      {
		     // printError(&_SciErr, 0);
		      return _SciErr;
	      }
	 
	
	
	// nr_class
	//rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
	//rhs[out_id] = (double *)MALLOC(1*1*sizeof(double));
	ptr = (rhs[out_id]);
	ptr[0] = model->nr_class;


	 // total SV
	  out_id++;
	 _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 5, 1, 1, &rhs[out_id]);
           if(_SciErr.iErr)
	      {
		      //printError(&_SciErr, 0);
		      return _SciErr;
	      }
	 
	
	// total SV
	//rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
	//rhs[out_id] = (double *)MALLOC(1*1*sizeof(double));
	ptr = (rhs[out_id]);
	ptr[0] = model->l;


	 // rho
	 out_id++;
	 n = model->nr_class*(model->nr_class-1)/2;
	 _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 6, n, 1, &rhs[out_id]);
           if(_SciErr.iErr)
	      {
		      //printError(&_SciErr, 0);
		      return _SciErr;
	      }
	// rho
	n = model->nr_class*(model->nr_class-1)/2;
	//rhs[out_id] = mxCreateDoubleMatrix(n, 1, mxREAL);
	//rhs[out_id] = (double *)MALLOC(n*1*sizeof(double));
	ptr = (rhs[out_id]);
	for(i = 0; i < n; i++)
		ptr[i] = model->rho[i];
	
          
	 // Label
	  out_id++;
	  if (model->label){
	     _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 7, model->nr_class, 1, &rhs[out_id]);
           if(_SciErr.iErr)
	      {
		      //printError(&_SciErr, 0);
		      return _SciErr;
	      }
	  } else
	     _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 7, 0, 0, &rhs[out_id]);
	    
	
	
	
	// Label
	if(model->label)
	{
		//rhs[out_id] = mxCreateDoubleMatrix(model->nr_class, 1, mxREAL);
		//rhs[out_id] = (double *)MALLOC(model->nr_class*1*sizeof(double));
		ptr = (rhs[out_id]);
		for(i = 0; i < model->nr_class; i++)
			ptr[i] = model->label[i];
	}
	else
		//rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
	        rhs[out_id] = NULL;// (double *)MALLOC(0*0*sizeof(double));

	 // probA
 	  out_id++;
	  if (model->probA != NULL){
	    _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 8, n, 1, &rhs[out_id]);
	     if(_SciErr.iErr)
	{
		//printError(&_SciErr, 0);
		return _SciErr;
	}
	  } else
	     _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 8, 0, 0, &rhs[out_id]);
          

	// probA
	if(model->probA != NULL)
	{
		//rhs[out_id] = mxCreateDoubleMatrix(n, 1, mxREAL);
		//rhs[out_id] = (double *)MALLOC(n*1*sizeof(double));
		ptr = (rhs[out_id]);
		for(i = 0; i < n; i++)
			ptr[i] = model->probA[i];
	}
	else
		//rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
	        rhs[out_id] = NULL;
	
	 // probB
	  out_id++;
	  if (model->probB != NULL){
	    _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 9, n, 1, &rhs[out_id]);
	     if(_SciErr.iErr)
	{
		//printError(&_SciErr, 0);
		return _SciErr;
	}
	  } else
	     _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 9, 0, 0, &rhs[out_id]);
	  

	// probB
	if(model->probB != NULL)
	{
		//rhs[out_id] = mxCreateDoubleMatrix(n, 1, mxREAL);
		//rhs[out_id] = (double *)MALLOC(n*1*sizeof(double));
		ptr = (rhs[out_id]);
		for(i = 0; i < n; i++)
			ptr[i] = model->probB[i];
	}
	else
		//rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
	        rhs[out_id] = NULL;

           // nSV
 	  out_id++;
	  if (model->nSV ){
	    _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 10, model->nr_class, 1, &rhs[out_id]);
	     if(_SciErr.iErr)
	{
		//printError(&_SciErr, 0);
		return _SciErr;
	}
	  } else
	     _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 10, 0, 0, &rhs[out_id]);
	  
	// nSV
	if(model->nSV)
	{
		//rhs[out_id] = mxCreateDoubleMatrix(model->nr_class, 1, mxREAL);
		//rhs[out_id] = (double *)MALLOC(model->nr_class*1*sizeof(double));
		ptr = (rhs[out_id]);
		for(i = 0; i < model->nr_class; i++)
			ptr[i] = model->nSV[i];
	}
	else
		//rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
	        rhs[out_id] = NULL;
	
	
         // sv_coef
	 out_id++;
	
	  _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 11, model->l, model->nr_class-1, &rhs[out_id]);
	   if(_SciErr.iErr)
	{
		//printError(&_SciErr, 0);
		return _SciErr;
	}
	
	// sv_coef
	//rhs[out_id] = mxCreateDoubleMatrix(model->l, model->nr_class-1, mxREAL);
	//rhs[out_id] = (double *)MALLOC(model->l*(model->nr_class-1)*sizeof(double));
	ptr = (rhs[out_id]);
	for(i = 0; i < model->nr_class-1; i++)
		for(j = 0; j < model->l; j++)
			ptr[(i*(model->l))+j] = model->sv_coef[i][j];


       
        // SVs
        
               out_id++;
	       
if (USE_SPARSE_IN_STRUCT==1) {	       
	      
                //mxArray *pprhs[1], *pplhs[1];   

                if(model->param.kernel_type == PRECOMPUTED)
		{
			nonzero_element = model->l;
			num_of_feature = 1;
		}
		else
		{
			nonzero_element = 0;
			for(i = 0; i < model->l; i++) {
				j = 0;
				while(model->SV[i][j].index != -1) 
				{
					nonzero_element++;
					j++;
				}
			}
		}

		// First, allocate a new sparse matrix
  ConstrMat      = (SciSparse *)MALLOC(1*sizeof(SciSparse));

  ConstrMat->n    = num_of_feature;
  ConstrMat->m    = model->l;
  ConstrMat->it   = 0;
  ConstrMat->nel  = nonzero_element;
  ConstrMat->mnel = (int *)MALLOC(model->l*sizeof(int));
  ConstrMat->icol = (int *)MALLOC(nonzero_element*sizeof(int));
  ConstrMat->R    = (double *)MALLOC(nonzero_element*sizeof(double));

  if ((ConstrMat==(SciSparse *)0) || (ConstrMat->mnel==NULL) || (ConstrMat->R==NULL))
    {
      Scierror(999,  "error while allocating the sparse\n");
      return _SciErr;
    }

		// SV in column, easier accessing
		//rhs[out_id] = mxCreateSparse(num_of_feature, model->l, nonzero_element, mxREAL);
		//rhs[out_id] = (double *)MALLOC(nonzero_element*1*sizeof(double));
                //ir = mxGetIr(rhs[out_id]);
		//ir = (int *)MALLOC((model->l)*1*sizeof(int));
                //jc = mxGetJc(rhs[out_id]);
		//jc = (int *)MALLOC(nonzero_element*1*sizeof(int));
		printf("nonzero %d, num_of_feature %d, model->l %d\n",nonzero_element,num_of_feature,model->l);
                //ptr = (rhs[out_id]);
                //jc[0]  = 1;   
		ConstrMat->icol[0]=1;
		jc_index = 0; 
		ir_index = 0;
                for(i = 0;i < model->l; i++)
                {
                        if(model->param.kernel_type == PRECOMPUTED)
			{
				// make a (1 x model->l) matrix
				//ir[ir_index] = 1; 
				ConstrMat->mnel[ir_index] = 1;
				//ptr[ir_index] = model->SV[i][0].value;
				ConstrMat->R[ir_index] = model->SV[i][0].value;
				ConstrMat->icol[ir_index]=ir_index;
				ir_index++;
				//jc[i] = ir_index + 1;
			}
			else
			{
				int x_index = 0;
				while (model->SV[i][x_index].index != -1)
				{
					//jc[jc_index] = model->SV[i][x_index].index ; 
					ConstrMat->icol[jc_index]=model->SV[i][x_index].index ; 
					//ptr[jc_index] = model->SV[i][x_index].value;
					ConstrMat->R[jc_index]= model->SV[i][x_index].value;
					jc_index++, x_index++;
					//if (i<=1)
					//   printf("i %d, x_index %d, jc_index %d, ir[ir_index] %d\n",i,x_index,jc_index,ir[ir_index]);
					
				}
				
				 // ir[i] =  x_index;
				  ConstrMat->mnel[i] = x_index;
				  //printf("ir = %d\n",ir[i]);
			}
                }
                // transpose back to SV in row
//                 pprhs[0] = rhs[out_id];
//                 if(mexCallSCILAB(1, pplhs, 1, pprhs, "transpose"))
//                         return "cannot transpose SV matrix";
//                 rhs[out_id] = pplhs[0];
              
	

	// Create a struct matrix contains NUM_OF_RETURN_FIELD fields 
	//return_model = mxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, field_names);
	
	   // SVs
	   
	   
	  //_SciErr = createSparseMatrixInList(pvApiCtx, Rhs + 1, piAddr, out_id+3, model->l, num_of_feature  , nonzero_element, ir, jc, rhs[out_id]);
	 //allocSparseMatrix(pvApiCtx, Rhs + 2, model->l, num_of_feature  , nonzero_element, &ir, &jc, &rhs[out_id]);
	
	 // _SciErr = createSparseMatrixInList(pvApiCtx, Rhs + 1, piAddr, out_id+3, 3, 10, 4, piNbItemRow, piColPos, pdblSReal);
	  ////_SciErr = createSparseMatrixInList(pvApiCtx, Rhs + 1, piAddr, out_id+2, 3, 10, 4, piNbItemRow, piColPos, pdblSReal);
	  
	   _SciErr = createSparseMatrixInList(pvApiCtx, Rhs+1, piAddr, 12,  ConstrMat->m, ConstrMat->n, ConstrMat->nel,  ConstrMat->mnel, ConstrMat->icol, ConstrMat->R); 
	 if(_SciErr.iErr)
	{
		//printError(&_SciErr, 0);
		return _SciErr;
	}
	 
	
	  
	 	if (ConstrMat->mnel) FREE(ConstrMat->mnel);
  if (ConstrMat->icol) FREE(ConstrMat->icol);
  if (ConstrMat->R)    FREE(ConstrMat->R);
  if (ConstrMat)       FREE(ConstrMat);
	 
} else {
  
  if(model->param.kernel_type == PRECOMPUTED)
  {
			nonzero_element = model->l;
			num_of_feature = 1;
    
  }
  
    _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 12, model->l, num_of_feature, &rhs[out_id]);
	   if(_SciErr.iErr)
	{
		//printError(&_SciErr, 0);
		return _SciErr;
	}
	
  ptr = (rhs[out_id]);
  for(i = 0;i < model->l; i++)
            for (j=0;j<num_of_feature;j++)
	        ptr[(i*num_of_feature)+j]=0;
  
                jc_index = 0; 
		ir_index = 0;
                for(i = 0;i < model->l; i++)
                {
                        if(model->param.kernel_type == PRECOMPUTED)
			{
				
				//ptr[ir_index] = model->SV[i][0].value;
				ptr[ir_index] = model->SV[i][0].value;
				
				ir_index=ir_index+num_of_feature;
				//jc[i] = ir_index + 1;
			}
			else
			{
				int x_index = 0;
				while (model->SV[i][x_index].index != -1)
				{
					//jc[jc_index] = model->SV[i][x_index].index ; 
					ptr[(model->SV[i][x_index].index-1)*model->l+i]=model->SV[i][x_index].value;
					
					//ptr[(num_of_feature*ir_index)+(model->SV[i][x_index].index-1)]=model->SV[i][x_index].value;
					
					//ptr[jc_index] = model->SV[i][x_index].value;
					//ConstrMat->R[jc_index]= model->SV[i][x_index].value;
					
					//if (i<=1)
					//  printf("i %d, x_index %d, ptr_ind %d, sv %f\n",i,x_index,((model->SV[i][x_index].index-1)*model->l+i),model->SV[i][x_index].value);
					jc_index++, x_index++;
				}
				ir_index++;
				 // ir[i] =  x_index;
				  //ConstrMat->mnel[i] = x_index;
				  //printf("ir = %d\n",ir[i]);
			}
                }

}

// 	 FREE(rhs[out_id]);
// 	 FREE(ir);
// 	 FREE(jc);
// 	 FREE(rhs);
	 
	 /*
	
	 
	
	// Fill struct matrix with input arguments 
 	for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
 		//mxSetField(return_model,0,field_names[i],mxDuplicateArray(rhs[i]));
 	   if (rhs[i]!=NULL )
 	        FREE(rhs[i]);
	   
 	FREE(rhs);
   
    if (ir!=NULL )
 	FREE(ir);
       if (jc!=NULL )
 	FREE(jc);
       if (dims!=NULL)
 	 FREE(dims);
	// return 
	//plhs[0] = return_model;
	//mxFree(rhs);
       
       */
       
                            
       
       
       

        return _SciErr;
}



struct svm_model *scilab_matrix_to_model(int *scilab_struct, const char **msg)
{
        int i, j,jj,n, num_of_fields;
        double *ptr;
	int id = 0;
	struct svm_node *x_space;
        struct svm_model *model;
       // mxArray **rhs;
        SciErr _SciErr;
	int iRows           = 0;
	int iCols           = 0;
	 int sr, sc, elements;
         int num_samples;
         int *ir, *jc;
	 int x_index = 0;
	
		
	_SciErr = getListItemNumber(pvApiCtx, scilab_struct, &num_of_fields);
	if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	
        //num_of_fields = mxGetNumberOfFields(scilab_struct);
        if(num_of_fields != NUM_OF_RETURN_FIELD+2) 
        {
                *msg = "number of return field is not correct";
		return NULL;
	}
        //rhs = (double **) MALLOC(sizeof(double *)*num_of_fields);

        //for(i=0;i<num_of_fields;i++)
        //        rhs[i] = mxGetFieldByNumber(scilab_struct, 0, i);

        model = Malloc(struct svm_model, 1);
        model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->label = NULL;
	model->nSV = NULL;
	model->free_sv = 1; // XXX

	//ptr = mxGetPr(rhs[id]);
	
	_SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 3, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	model->param.svm_type = (int)ptr[0];
	model->param.kernel_type  = (int)ptr[1];
	model->param.degree	  = (int)ptr[2];
	model->param.gamma	  = ptr[3];
	model->param.coef0	  = ptr[4];
	id++;

	
	
	//ptr = mxGetPr(rhs[id]);
	_SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 4, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	model->nr_class = (int)ptr[0];
	id++;

	//ptr = mxGetPr(rhs[id]);
	_SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 5, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	model->l = (int)ptr[0];
	id++;

	// rho
	n = model->nr_class * (model->nr_class-1)/2;
	model->rho = (double*) malloc(n*sizeof(double));
	//ptr = mxGetPr(rhs[id]);
	_SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 6, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	for(i=0;i<n;i++)
		model->rho[i] = ptr[i];
	id++;

	// label
	_SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 7, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	if(iRows*iCols>0)
	{
		model->label = (int*) malloc(model->nr_class*sizeof(int));
		//ptr = mxGetPr(rhs[id]);
		for(i=0;i<model->nr_class;i++)
			model->label[i] = (int)ptr[i];
	}
	id++;

	// probA
	_SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 8, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	if(iRows*iCols>0)
	{
		model->probA = (double*) malloc(n*sizeof(double));
		//ptr = mxGetPr(rhs[id]);
		for(i=0;i<n;i++)
			model->probA[i] = ptr[i];
	}
	id++;

	// probB
	_SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct,9, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	if(iRows*iCols>0)
	{
		model->probB = (double*) malloc(n*sizeof(double));
		//ptr = mxGetPr(rhs[id]);
		for(i=0;i<n;i++)
			model->probB[i] = ptr[i];
	}
	id++;

	// nSV
	_SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 10, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	if(iRows*iCols>0)
	{
		model->nSV = (int*) malloc(model->nr_class*sizeof(int));
		//ptr = mxGetPr(rhs[id]);
		for(i=0;i<model->nr_class;i++)
			model->nSV[i] = (int)ptr[i];
	}
	id++;

	// sv_coef
	//ptr = mxGetPr(rhs[id]);
	_SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 11, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	
	model->sv_coef = (double**) malloc((model->nr_class-1)*sizeof(double));
	for( i=0 ; i< model->nr_class -1 ; i++ )
		model->sv_coef[i] = (double*) malloc((model->l)*sizeof(double));
	for(i = 0; i < model->nr_class - 1; i++)
		for(j = 0; j < model->l; j++)
			model->sv_coef[i][j] = ptr[i*(model->l)+j];
	id++;

	// SV
	if (USE_SPARSE_IN_STRUCT==1) {
	 _SciErr = getSparseMatrixInList(pvApiCtx, scilab_struct, 12, &sr, &sc, &num_samples, &ir, &jc, &ptr);
	     if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
        //{
               
//                mxArray *pprhs[1], *pplhs[1];

                // transpose SV
//                 pprhs[0] = rhs[id];
//                 if(mexCallSCILAB(1, pplhs, 1, pprhs, "transpose")) 
//                 {
//                         svm_free_and_destroy_model(&model);
//                         *msg = "cannot transpose SV matrix";
//                         return NULL;
//                 }
//                 rhs[id] = pplhs[0];

//                 sr = mxGetN(rhs[id]);
//                 sc = mxGetM(rhs[id]);
// 
//                 ptr = mxGetPr(rhs[id]);
//                 ir = mxGetIr(rhs[id]);
//                 jc = mxGetJc(rhs[id]);
// 
//                 num_samples = mxGetNzmax(rhs[id]);

                elements = num_samples + sr;

		model->SV = (struct svm_node **) malloc(sr * sizeof(struct svm_node *));
		x_space = (struct svm_node *)malloc(elements * sizeof(struct svm_node));
                 
// 		printf("sr %d, num_samples %d\n",sr,num_samples);
                // SV is in column
		jj=0; 
                for(i=0;i<sr;i++)
                {
                        int low = 0, high = ir[i];
                        x_index = 0;
                        model->SV[i] = &x_space[jj+i];
/*			if (i<=3)
			   printf("i: %d, jj %d \n",i,jj);
  */                      for(j=low;j<high;j++)
                        {
                                model->SV[i][x_index].index = jc[jj]; 
                                model->SV[i][x_index].value = ptr[jj];
/*				if (i<=2)
				   printf("i %d, x_index %d jc %d ptr %f\n",i,x_index,model->SV[i][x_index].index,model->SV[i][x_index].value);
 */                               x_index++;
				jj++;
                        }
//                         if (i<=2)
// 				   printf("i %d, x_index %d, index = -1\n",i,x_index);
			model->SV[i][x_index].index = -1;
			
			
		}
	} else {
	  _SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 12, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	
	
	                num_samples = 0;
			for(i = 0; i < (iRows*iCols); i++) {
				if (ptr[i]!=0)
					num_samples++;
			}
	
	        sr=iRows;
	        elements = num_samples + sr;
  
// 		printf("sr %d, num_samples %d, iCols %d\n",sr,num_samples,iCols);
		model->SV = (struct svm_node **) malloc(sr * sizeof(struct svm_node *));
		x_space = (struct svm_node *)malloc(elements * sizeof(struct svm_node));

                // SV is in column
		jj=0; 
                for(i=0;i<sr;i++)
                {
                        int low = 0, high = iCols;
                        x_index = 0;
                        model->SV[i] = &x_space[jj+i];
// 			if (i<=3)
// 			   printf("jj %d, i%d\n",jj,i);
                        for(j=low;j<high;j++)
                        {
			  if (ptr[(j*iRows)+i]!=0) {
                                model->SV[i][x_index].index = j+1; 
                                model->SV[i][x_index].value = ptr[(j*iRows)+i];
// 				if (i<=2)
// 				   printf("i %d, x_index %d j %d index %d ptr %f\n",i,x_index,j,model->SV[i][x_index].index,model->SV[i][x_index].value);
                                x_index++;
				jj++;
				
			  }
                        }
//                          if (i<=2)
// 				   printf("i %d, x_index %d, index = -1\n",i,x_index);
			model->SV[i][x_index].index = -1;
			
			
		}
	  
	  
	}
		id++;
		
		 //printf("i %d, x_index %d jc %d ptr %f\n",2,0,model->SV[2][0].index,model->SV[2][0].value);
	//}
	//mxFree(rhs);

	return model;
}
