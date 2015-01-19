//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2000-2011 Chih-Chung Chang and Chih-Jen Lin
// Modifiered 2011 by Holger Nahrstaedt and Yann Collette
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


#include <api_scilab.h>
// #define __USE_DEPRECATED_STACK_FUNCTIONS__
// #include <stack-c.h>
#include <sciprint.h>
#include <MALLOC.h>
#include <Scierror.h>



// #define DEBUG

int libsvmwrite(const char *filename, int * label_vec, int * instance_mat)
{
  FILE *fp = fopen(filename,"w"); 
  int i, k, low, high, r_samples, c_samples, r_labels, c_labels, index;
  int label_vector_row_num, type, tmp, elements;
  int * samples_piNbItemRow = NULL, * samples_piColPos = NULL;
  double *samples = NULL, *labels = NULL;
  SciErr _SciErr;

  if (fp==NULL)
    {
      Scierror (999,"can't open output file %s\n",filename);			
      return -1;
    }

  // the number of instance
  _SciErr = getVarType(pvApiCtx, instance_mat, &type);
  if (type==sci_matrix)
    {
#ifdef DEBUG
      printf("DEBUG: full matrix\n");
#endif
      _SciErr = getMatrixOfDouble(pvApiCtx, instance_mat, &r_samples, &c_samples, &samples);
      _SciErr = getMatrixOfDouble(pvApiCtx, label_vec, &r_labels, &c_labels, &labels);
#ifdef DEBUG
      printf("DEBUG: inst:   row = %d col = %d\n", r_samples, c_samples);
      printf("DEBUG: labels: row = %d col = %d\n", r_labels,  c_labels);
#endif
      if (r_labels!=r_samples)
        {
         Scierror (999,"Length of label vector does not match # of instances.\n");
          fclose(fp);
          return -1;
        }

      for(i=0;i<r_labels;i++)
        {
          fprintf(fp,"%g", labels[i]);
#ifdef DEBUG
          printf("DEBUG: line %d - %g", i+1, labels[i]);
#endif
          for(k=0;k<c_samples;k++)
            {
              fprintf(fp," %ld:%g", k+1, samples[i + k*r_samples]);		
#ifdef DEBUG
              printf(" %ld:%g", k+1, samples[i + k*r_samples]);
#endif
            }
          
          fprintf(fp,"\n");
#ifdef DEBUG
          printf("\n");
#endif
        }
      fclose(fp);
    }
  else if (type==sci_sparse)
    {
#ifdef DEBUG
      printf("DEBUG: sparse matrix\n");
#endif
      _SciErr = getSparseMatrix(pvApiCtx, instance_mat, &r_samples, &c_samples, &elements, &samples_piNbItemRow, &samples_piColPos, &samples);
      _SciErr = getMatrixOfDouble(pvApiCtx, label_vec, &r_labels, &c_labels, &labels);
      if (r_samples!=r_labels)
        {
          Scierror (999,"Length of label vector does not match # of instances.\n");
          fclose(fp);
          return -1;
        }

      index = 0;
      for(i=0;i<r_labels;i++)
        {
          fprintf(fp,"%g", labels[i]);
#ifdef DEBUG
          printf("DEBUG: %g", labels[i]);
#endif
          
          for(k=0;k<samples_piNbItemRow[i];k++)
            {
              fprintf(fp," %ld:%g", samples_piColPos[index], samples[index]);
#ifdef DEBUG
              printf(" %ld:%g", samples_piColPos[index], samples[index]);
#endif
              index++;
            }
          
          fprintf(fp,"\n");
#ifdef DEBUG
          printf("\n");
#endif
        }
      fclose(fp);
    }
  
  return 0;
}

int sci_libsvmwrite(char * fname)
{
  int * p_address_filename = NULL;
  int * p_label_vector = NULL;
  int * p_instance_matrix = NULL;
  int type;
  char * filename = NULL;
  SciErr _SciErr;

  // Transform the input Matrix to libsvm format
  if (nbInputArgument(pvApiCtx) == 3)
    {
      _SciErr = getVarAddressFromPosition(pvApiCtx, 2, &p_label_vector);
      _SciErr = getVarType(pvApiCtx, p_label_vector, &type);

      if (type!=sci_matrix && type!=sci_sparse)
        {
         Scierror (999,"Error: label vector must be double\n");			
          return 0;
        }
      
      _SciErr = getVarAddressFromPosition(pvApiCtx, 3, &p_instance_matrix);
      _SciErr = getVarType(pvApiCtx, p_instance_matrix, &type);

      if (type!=sci_matrix && type!=sci_sparse)
        {
          Scierror (999,"Error: instance matrix must be double\n");			
          return 0;
        }

      _SciErr = getVarAddressFromPosition(pvApiCtx, 1, &p_address_filename);
      getAllocatedSingleString(pvApiCtx, p_address_filename, &filename);
      
      if (filename == NULL)
        {
          Scierror(999, "Error: filename is NULL\n");
          return 0;
        }

      libsvmwrite(filename, p_label_vector, p_instance_matrix);
      freeAllocatedSingleString(filename);
    }
  else
    {
      Scierror (999,"Usage: libsvmwrite('filename', label_vector, instance_matrix);\n");
    }

      /* This function put on scilab stack, the lhs variable
  which are at the position lhs(i) on calling stack */
  /* You need to add PutLhsVar here because WITHOUT_ADD_PUTLHSVAR 
  was defined and equal to %t */
  /* without this, you do not need to add PutLhsVar here */
  ReturnArguments(pvApiCtx);
    
  return 0;
}
