#include <stdlib.h>
#include <string.h>
#include "linear.h"

#include <api_scilab.h>
#include <stack-c.h>
#include <sciprint.h>
#include <MALLOC.h>
#include <Scierror.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#define NUM_OF_RETURN_FIELD 6

static const char *field_names[] = {
        "st",
	"dims",
	"Parameters",
	"nr_class",
	"nr_feature",
	"bias",
	"Label",
	"w"
};

const char *linear_model_to_scilab_structure( struct model *model_)
{
	int i;
	int nr_w;
	double *ptr,*dims;
	double *return_model, **rhs;
	int out_id = 0;
	int n, w_size;
	int *piAddr             = NULL;
	SciErr _SciErr;
	//rhs = (mxArray **)mxMalloc(sizeof(mxArray *)*NUM_OF_RETURN_FIELD);
        rhs = (double **)MALLOC(sizeof(double *)*NUM_OF_RETURN_FIELD);

	
	_SciErr = createMList(pvApiCtx,Rhs + 1,NUM_OF_RETURN_FIELD+2, &piAddr);
      if(_SciErr.iErr)
	      {
		      printError(&_SciErr, 0);
		      return NULL;
	      }
	
	_SciErr = createMatrixOfStringInList(pvApiCtx, Rhs + 1, piAddr, 1, NUM_OF_RETURN_FIELD+2, 1, field_names);
           if(_SciErr.iErr)
	      {
		      printError(&_SciErr, 0);
		      return NULL;
	      }
	      
	  _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 2, 1, 2 , &dims);
           if(_SciErr.iErr)
	      {
		      printError(&_SciErr, 0);
		      return NULL;
	      }
	       dims[0]=1;
	       dims[1]=1;
	       
	       
	// Parameters
	// for now, only solver_type is needed
	//rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
        _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 3, 1, 1, &rhs[out_id]);
           if(_SciErr.iErr)
	      {
		      printError(&_SciErr, 0);
		      return NULL;
	      }
	ptr = (rhs[out_id]);
	ptr[0] = model_->param.solver_type;
	out_id++;

	// nr_class
	_SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 4, 1, 1, &rhs[out_id]);
           if(_SciErr.iErr)
	      {
		      printError(&_SciErr, 0);
		      return NULL;
	      }
	//rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = (rhs[out_id]);
	ptr[0] = model_->nr_class;
	out_id++;

	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	// nr_feature
		_SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 5, 1, 1, &rhs[out_id]);
           if(_SciErr.iErr)
	      {
		      printError(&_SciErr, 0);
		      return NULL;
	      }
	//rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = (rhs[out_id]);
	ptr[0] = model_->nr_feature;
	out_id++;

	// bias
	_SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 6, 1, 1, &rhs[out_id]);
           if(_SciErr.iErr)
	      {
		      printError(&_SciErr, 0);
		      return NULL;
	      }
	//rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = (rhs[out_id]);
	ptr[0] = model_->bias;
	out_id++;

	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;

	w_size = n;
	// Label
	if(model_->label)
	{
	    _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 7, model_->nr_class, 1, &rhs[out_id]);
           if(_SciErr.iErr)
	      {
		      printError(&_SciErr, 0);
		      return NULL;
	      }
		//rhs[out_id] = mxCreateDoubleMatrix(model_->nr_class, 1, mxREAL);
		ptr = (rhs[out_id]);
		for(i = 0; i < model_->nr_class; i++)
			ptr[i] = model_->label[i];
	}
	else
		//rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
	      _SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr, 7, 0, 0, &rhs[out_id]);
	out_id++;

	// w
	_SciErr = allocMatrixOfDoubleInList(pvApiCtx, Rhs + 1, piAddr,8, nr_w, w_size, &rhs[out_id]);
           if(_SciErr.iErr)
	      {
		      printError(&_SciErr, 0);
		      return NULL;
	      }
	//rhs[out_id] = mxCreateDoubleMatrix(nr_w, w_size, mxREAL);
	ptr = (rhs[out_id]);
	for(i = 0; i < w_size*nr_w; i++)
		ptr[i]=model_->w[i];
	out_id++;

	/* Create a struct matrix contains NUM_OF_RETURN_FIELD fields */
	//return_model = mxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, field_names);

	/* Fill struct matrix with input arguments */
	//for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
	//	mxSetField(return_model,0,field_names[i],mxDuplicateArray(rhs[i]));
	/* return */
	//plhs[0] = return_model;
	//mxFree(rhs);
	
	  LhsVar(1) = Rhs+1; 
  /* This function put on scilab stack, the lhs variable
  which are at the position lhs(i) on calling stack */
  /* You need to add PutLhsVar here because WITHOUT_ADD_PUTLHSVAR 
  was defined and equal to %t */
  /* without this, you do not need to add PutLhsVar here */
       PutLhsVar();
	return NULL;
}

const char *scilab_matrix_to_linear_model(int *scilab_struct, struct model *model_)
{
	int i, num_of_fields;
	int nr_w;
	double *ptr;
	int id = 0;
	int n, w_size;
	//double **rhs;
	 SciErr _SciErr;
	int iRows           = 0;
	int iCols           = 0;

// 	num_of_fields = mxGetNumberOfFields(matlab_struct);
// 	rhs = (mxArray **) mxMalloc(sizeof(mxArray *)*num_of_fields);
// 
// 	for(i=0;i<num_of_fields;i++)
// 		rhs[i] = mxGetFieldByNumber(matlab_struct, 0, i);
	
	_SciErr = getListItemNumber(pvApiCtx, scilab_struct, &num_of_fields);
	if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	

	model_->nr_class=0;
	nr_w=0;
	model_->nr_feature=0;
	model_->w=NULL;
	model_->label=NULL;

	// Parameters
	_SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 3, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	//ptr = mxGetPr(rhs[id]);
	model_->param.solver_type = (int)ptr[0];
	id++;

	// nr_class
	_SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 4, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	//ptr = mxGetPr(rhs[id]);
	model_->nr_class = (int)ptr[0];
	id++;

	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	// nr_feature
		_SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 5, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	//ptr = mxGetPr(rhs[id]);
	model_->nr_feature = (int)ptr[0];
	id++;

	// bias
	_SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 6, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	//ptr = mxGetPr(rhs[id]);
	model_->bias = (int)ptr[0];
	id++;

	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	w_size = n;
	
	
        _SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 7, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	//ptr = mxGetPr(rhs[id]);
	model_->label=Malloc(int, model_->nr_class);
	for(i=0; i<model_->nr_class; i++)
		model_->label[i]=(int)ptr[i];
	id++;
	
        _SciErr = getMatrixOfDoubleInList(pvApiCtx, scilab_struct, 8, &iRows, &iCols, &ptr);
	    if(_SciErr.iErr)
	{
		printError(&_SciErr, 0);
		return NULL;
	}
	//ptr = mxGetPr(rhs[id]);
	model_->w=Malloc(double, w_size*nr_w);
	for(i = 0; i < w_size*nr_w; i++)
		model_->w[i]=ptr[i];
	id++;
	//mxFree(rhs);

	return NULL;
}

