#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <api_scilab.h>
#include <stack-c.h>
#include <sciprint.h>
#include <Scierror.h>
#include <MALLOC.h>


// #define DEBUG

void libsvmwrite(const char *filename, int * label_vec, int * instance_mat)
{
  FILE *fp = fopen(filename,"w+"); // w+ pour linux ?
  int i, k, low, high, r_samples, c_samples, r_labels, c_labels, index;
  int label_vector_row_num, type, tmp, elements;
  int * samples_piNbItemRow = NULL, * samples_piColPos = NULL;
  double *samples = NULL, *labels = NULL;
  SciErr _SciErr;

  if (fp==NULL)
    {
      sciprint("can't open output file %s\n",filename);			
      return;
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
          sciprint("Length of label vector does not match # of instances.\n");
          fclose(fp);
          return;
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
          sciprint("Length of label vector does not match # of instances.\n");
          fclose(fp);
          return;
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
  
  return;
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
  if (Rhs == 3)
    {
      _SciErr = getVarAddressFromPosition(pvApiCtx, 2, &p_label_vector);
      _SciErr = getVarType(pvApiCtx, p_label_vector, &type);

      if (type!=sci_matrix && type!=sci_sparse)
        {
          sciprint("Error: label vector must be double\n");			
          return 0;
        }
      
      _SciErr = getVarAddressFromPosition(pvApiCtx, 3, &p_instance_matrix);
      _SciErr = getVarType(pvApiCtx, p_instance_matrix, &type);

      if (type!=sci_matrix && type!=sci_sparse)
        {
          sciprint("Error: instance matrix must be double\n");			
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
      sciprint("Usage: libsvmwrite('filename', label_vector, instance_matrix);\n");
    }

  return 0;
}
