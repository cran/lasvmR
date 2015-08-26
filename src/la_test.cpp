// -*- mode: c++; c-file-style: "stroustrup"; -*-


#include <stdio.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <cctype>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <Rcpp.h>

using namespace std;
using namespace Rcpp;

#include "vector.h"

#define LINEAR  0
#define POLY    1
#define RBF     2
#define SIGMOID 3 

extern const char *kernel_type_table[];// = {"linear","polynomial","rbf","sigmoid"};

class ID // class to hold split file indices and labels
{
public:
	int x;
	int y;
    ID() : x(0), y(0) {}
    ID(int x1,int y1) : x(x1), y(y1) {}
};
// IDs will be sorted by index, not by label.
bool operator<(const ID& x, const ID& y);

extern int m;
int msv;                         // training and test set sizes
extern vector <lasvm_sparsevector_t*> X; // feature vectors for test set
vector <lasvm_sparsevector_t*> Xsv;// feature vectors for SVs
extern vector <int> Y;                   // labels
extern vector <double> alpha;            // alpha_i, SV weights
extern double b0;                        // threshold
extern int use_b0;                     // use threshold via constraint \sum a_i y_i =0
extern int kernel_type;              // LINEAR, POLY, RBF or SIGMOID kernels
extern double degree;
extern double kgamma;
extern double coef0; // kernel params
extern vector <double> x_square;         // norms of input vectors, used for RBF
vector <double> xsv_square;        // norms of test vectors, used for RBF
extern char split_file_name[1024];         // filename for the splits
extern int binary_files;
extern vector <ID> splits;             
extern int max_index;

/* INIT THESE
 * extern int kernel_type=RBF;              // LINEAR, POLY, RBF or SIGMOID kernels
 * extern char split_file_name[1024]="\0";         // filename for the splits
 * e xtern int binary_files=0;*
 * extern int max_index;
 * extern int use_b0;                     // use threshold via constraint \sum a_i y_i =0
extern int kernel_type=RBF;              // LINEAR, POLY, RBF or SIGMOID kernels
extern double degree;
extern double kgamma;
extern double coef0; // kernel params
*/



void exit_with_help();





int split_file_load(char *f);

int libsvm_load_data(char *filename);

int binary_load_data(char *filename);

void load_data_file(char *filename);


void libsvm_load_sv_data(FILE *fp)
// loads the same format as LIBSVM
{ 
	stop ("should never be called directly.");
}




int libsvm_load_model(const char *model_file_name)
{
	stop ("should never be called directly.");
	return 0; // make everyone happy
}


double kernel(int i, int j, void *kparam);




double predictKernel(int i, int j, void *kparam)
{
	double dot;
	dot=lasvm_sparsevector_dot_product(X[i],Xsv[j]);
	
	// sparse, linear kernel
	switch(kernel_type)
	{
		case LINEAR:
			return dot; 
		case POLY:
			return pow(kgamma*dot+coef0,degree);
		case RBF:
			return exp(-kgamma*(x_square[i]+xsv_square[j]-2*dot));    
		case SIGMOID:
			return tanh(kgamma*dot+coef0);    
	}
	return 0;
}  



void test(char *output_name)
{	
	stop ("should never be called directly.");
}


void la_test_parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name, char *output_file_name)
{
	stop ("should never be called directly.");
}



int la_test_main(int argc, char **argv)  
{
    
    char input_file_name[1024];
    char model_file_name[1024];
    char output_file_name[1024];
    la_test_parse_command_line(argc, argv, input_file_name, model_file_name, output_file_name);
     
    libsvm_load_model(model_file_name);// load model
    load_data_file(input_file_name); // load test data
    
    test(output_file_name);
	return (-1);
}


