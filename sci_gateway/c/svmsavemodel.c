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
#include "svm.h"

#include <api_scilab.h>
// #define __USE_DEPRECATED_STACK_FUNCTIONS__
// #include <stack-c.h>
#include <sciprint.h>
#include <Scierror.h>

#include "svm_model_scilab.h"

//#define DEBUG

void exit_with_help_savemodel() {
  Scierror(999, "Usage: libsvm_svmsavemodel(filename,model)\n"
                "Parameters:\n"
                "  model: SVM model structure from svmtrain.\n"
                "  filename:\n"
                "   text file output\n");
}

int sci_svmsavemodel(char *fname) {
  SciErr _SciErr;
  struct svm_model *model;
  int *p_model = NULL;
  int *p_option_string = NULL;
  int type, type3;
  const char *error_msg;
  char *fileName = NULL;

  if (nbInputArgument(pvApiCtx) > 2 || nbInputArgument(pvApiCtx) < 2) {
    exit_with_help_savemodel();
    return 0;
  }

  _SciErr = getVarAddressFromPosition(pvApiCtx, 2, &p_model);
  if (_SciErr.iErr) {
    printError(&_SciErr, 0);
    return 0;
  }
  _SciErr = getVarType(pvApiCtx, p_model, &type);
  if (_SciErr.iErr) {
    printError(&_SciErr, 0);
    return 0;
  }

  // parse options
  // char cmd[CMD_LEN], *argv[CMD_LEN/2];
  // mxGetString(prhs[3], cmd,  mxGetN(prhs[3]) + 1);
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

  if ((type == sci_mlist || type == sci_list) && type3 == sci_strings) {
    getAllocatedSingleString(pvApiCtx, p_option_string, &fileName);

#ifdef DEBUG
    printf("DEBUG: start\n");
#endif

    model = scilab_matrix_to_model(p_model, &error_msg);
    if (model == NULL) {
      Scierror(999, "Error: can't read model: %s\n", error_msg);
      return 0;
    }
#ifdef DEBUG
    printf("DEBUG: read model done\n");
#endif

    svm_save_model(fileName, model);

#ifdef DEBUG
    printf("DEBUG: check probability done\n");
#endif
    // destroy model
    svm_free_and_destroy_model(&model);
    if (fileName != NULL)
      freeAllocatedSingleString(fileName);

    ReturnArguments(pvApiCtx);
  } else {
    exit_with_help_savemodel();
  }

  return 0;
}
