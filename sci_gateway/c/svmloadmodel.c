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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
#include <malloc.h>
#include <Scierror.h>

#include "svm_model_scilab.h"


void exit_with_help_loadmodel() {
  Scierror(999, "Usage: model = libsvm_loadmodel(filename)\n"
                "Parameters:\n"
                "  model: SVM model structure from svmtrain.\n"
                "  filename:\n"
                "   text file output\n");
}


int sci_svmloadmodel(char *fname,void* pvApiCtx)

{
  SciErr _SciErr;
  const char *error_msg;
  int *p_label_vector = NULL;
  int *p_instance_matrix = NULL;
  int *p_option_string = NULL;
  int r_samples, c_samples;
  double *samples = NULL;
  int type, type3;
  char *option_string = NULL;
  struct svm_model *model;
  int nr_feat = 1;
  // fix random seed to have same results for each run
  // (for cross validation and probability estimation)

  // Transform the input Matrix to libsvm format
  if (nbInputArgument(pvApiCtx) == 1) {
    int err;

    _SciErr = getVarAddressFromPosition(pvApiCtx, 1, &p_option_string);
    if (_SciErr.iErr) {
      printError(&_SciErr, 0);
      return 0;
    }
    _SciErr = getVarType(pvApiCtx, p_option_string, &type3);
    if (_SciErr.iErr) {
      printError(&_SciErr, 0);
      return 0;
    }
    if (type3 == sci_strings) {
      getAllocatedSingleString(pvApiCtx, p_option_string, &option_string);
    }

    // const char *error_msg;
    model = svm_load_model(option_string);
    nr_feat = model->nr_class;

    _SciErr = model_to_scilab_structure(nr_feat, model,pvApiCtx);
    if (_SciErr.iErr) {
      printError(&_SciErr, 0);
      exit_with_help_loadmodel();
      return 0;
    }
    AssignOutputVariable(pvApiCtx, 1) = nbInputArgument(pvApiCtx) + 1;
    // AssignOutputVariable(pvApiCtx,2) = nbInputArgument(pvApiCtx)+2;
    /* This function put on scilab stack, the lhs variable
    which are at the position lhs(i) on calling stack */
    /* You need to add PutLhsVar here because WITHOUT_ADD_PUTLHSVAR
    was defined and equal to %t */
    /* without this, you do not need to add PutLhsVar here */
    ReturnArguments(pvApiCtx);

    if (option_string != NULL)
      freeAllocatedSingleString(option_string);
    // if(error_msg)
    //        sciprint("Error: can't convert libsvm model to matrix structure:
    //        %s\n", error_msg);
    svm_free_and_destroy_model(&model);

  } else {
    exit_with_help_loadmodel();
    return 0;
  }
}
