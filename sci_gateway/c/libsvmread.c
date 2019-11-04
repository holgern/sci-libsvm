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
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>

#include <api_scilab.h>
// #define __USE_DEPRECATED_STACK_FUNCTIONS__
// #include <stack-c.h>
#include <sciprint.h>
//#include <malloc.h>
#include <Scierror.h>

#ifndef max
#define max(x, y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef min
#define min(x, y) (((x) < (y)) ? (x) : (y))
#endif

//#define DEBUG

static char *line;
static int max_line_len;

static char *readline(FILE *input) 
{
	int len;
	if (fgets(line, max_line_len, input) == NULL) 
	{
		#ifdef DEBUG
				sciprint("readline case 1\n");
		#endif
		return NULL;
	}

	while (strrchr(line, '\n') == NULL) 
	{
		max_line_len *= 2;
		line = (char *)realloc(line, max_line_len);
		len = (int)strlen(line);
		if (fgets(line + len, max_line_len - len, input) == NULL) 
		{
			#ifdef DEBUG
						sciprint("readline case 2\n");
			#endif
			break;
		}
	}
	return line;
}

// read in a problem (in libsvm format)
int read_problem(const char *filename, void* pvApiCtx) 
{
	int max_index, min_index, inst_max_index, i;
	long elements, k;
	FILE *fp = fopen(filename, "r");
	int l = 0;
	char *endptr = NULL;
	int *samples_piNbItemRow = NULL, *samples_piColPos = NULL;
	double *labels = NULL, *samples = NULL;
	SciErr _SciErr;

	if (fp == NULL) 
	{
		Scierror(999, "can't open input file %s\n", filename);
		return -1;
	}

	max_line_len = 1024;
	line = (char *)MALLOC(max_line_len * sizeof(char));

	max_index = 0;
	min_index = 1; // our index starts from 1
	elements = 0;

	while (readline(fp) != NULL) 
	{
		char *idx, *val;
		// features
		int index = 0;
		#ifdef DEBUG
				sciprint("preprocess line = %s\n", line);
		#endif
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed // kernel has <index> start from 0
		strtok(line, " \t"); // label
		while (1) 
		{
			idx = strtok(NULL, ":"); // index:value
			val = strtok(NULL, " \t");
			if (val == NULL) 
			{
				#ifdef DEBUG
						sciprint("DEBUG: break called\n");
				#endif
				break;
			}
			errno = 0;
			index = (int)strtol(idx, &endptr, 10);
			if (endptr == idx || errno != 0 || *endptr != '\0' || index <= inst_max_index) 
			{
				Scierror(999, "Wrong input format at line %d\n", l + 1);
				return -1;
			}
			else 
			{
				inst_max_index = index;
			}
			min_index = min(min_index, index);
			elements++;
		}
		max_index = max(max_index, inst_max_index);
		l++;
	}
	rewind(fp);

	#ifdef DEBUG
		sciprint("max_index = %d, min_index = %d elements = %d\n", max_index, min_index, elements);
	#endif

	// Add in MALLOC - Tan CL 25-Aug-2019
	samples_piNbItemRow = (int *)MALLOC(l * sizeof(int));
	samples_piColPos = (int *)MALLOC(elements * sizeof(int));
	labels = (double *)MALLOC(elements * sizeof(double));
	samples = (double *)MALLOC(elements * sizeof(double));

	// Lines for allocSparseMatrix have been replace with createSparseMatrix at the bottom of loops
	


	k = 0;
	for (i = 0; i < l; i++) 
	{
		char *idx, *val, *label;
		readline(fp);
		label = strtok(line, " \t");
		labels[i] = (double)strtod(label, &endptr);
		#ifdef DEBUG
				sciprint("DEBUG: labels[%d] = %f - \n", i, labels[i]);
		#endif
		if (endptr == label) 
		{
			Scierror(999, "Wrong input format at line %d\n", i + 1);
			return -1;
		}

		// features
		samples_piNbItemRow[i] = 0;
		while (1) {
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");
			if (val == NULL)
				break;

			samples_piColPos[k] = (int)(strtol(idx, &endptr, 10) - min_index) + 1; // precomputed kernel has <index> start from 0
			errno = 0;
			samples[k] = (double)strtod(val, &endptr);
			#ifdef DEBUG
						sciprint(" %d:%f \n", samples_piColPos[k], samples[k]);
			#endif
			samples_piNbItemRow[i]++;
			if (endptr == val || errno != 0 ||(*endptr != '\0' && !isspace(*endptr))) 
			{
				Scierror(999, "Wrong input format at line %d\n", i + 1);
				return -1;
			}
			++k;
		}
	#ifdef DEBUG
			sciprint("\n");
	#endif
	}
	
	// y
	_SciErr = createMatrixOfDouble(pvApiCtx, nbInputArgument(pvApiCtx) + 1, l, 1, labels);

	// x^T
	if (min_index <= 0)
	{
	#ifdef DEBUG
			sciprint("case 1: col = %d, row = %d elements = %d\n", max_index - min_index + 1, l, elements);
	#endif
		_SciErr = createSparseMatrix(pvApiCtx, nbInputArgument(pvApiCtx) + 2, l, max_index - min_index + 1, elements, samples_piNbItemRow, samples_piColPos, samples);
	}
	else
	{
	#ifdef DEBUG
			sciprint("case 2: col = %d, row = %d elements = %d\n", max_index, l, elements);
	#endif
		_SciErr = createSparseMatrix(pvApiCtx, nbInputArgument(pvApiCtx) + 2, l, max_index, elements, samples_piNbItemRow, samples_piColPos, samples);
	}

	
	AssignOutputVariable(pvApiCtx, 1) = nbInputArgument(pvApiCtx) + 1;
	AssignOutputVariable(pvApiCtx, 2) = nbInputArgument(pvApiCtx) + 2;
	/* This function put on scilab stack, the lhs variable
	which are at the position lhs(i) on calling stack */
	/* You need to add PutLhsVar here because WITHOUT_ADD_PUTLHSVAR
	was defined and equal to %t */
	/* without this, you do not need to add PutLhsVar here */
	ReturnArguments(pvApiCtx);
	fclose(fp);
	free(line);
	return 0;
}

int sci_libsvmread(char *fname, void* pvApiCtx) {
	int *p_address_filename = NULL, *p_out_data = NULL;
	char *filename = NULL;
	SciErr _SciErr;

	if (nbInputArgument(pvApiCtx) == 1) {
		_SciErr = getVarAddressFromPosition(pvApiCtx, 1, &p_address_filename);
		getAllocatedSingleString(pvApiCtx, p_address_filename, &filename);

		if (filename == NULL) {
			Scierror(999, "Error: filename is NULL\n");
			return 0;
		}

		read_problem(filename, pvApiCtx);

		freeAllocatedSingleString(filename);
	}
	else {
		Scierror(
			999,
			"Usage: [label_vector, instance_matrix] = libsvmread('filename');\n");
	}

	return 0;
}
