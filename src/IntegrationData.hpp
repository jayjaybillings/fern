#include "Network.hpp"

#ifndef IntegrationData_cuh
#define IntegrationData_cuh

struct Network;


struct IntegrationData
{
	unsigned short species;
	fern_real T9;
	fern_real H2O;
	fern_real M;
	fern_real Patm;
	fern_real zenith;
	fern_real alt;
	fern_real t_init;
	fern_real t_max;
	fern_real dt_init;
	fern_real rho;
	
	fern_real *Y;
	
	
	void loadAbundances(const char *filename);
	
	void allocate(unsigned short species);
	
	/**	Prints all scalars and vectors to stdout
		
		If the number of species is given, the values of the abundance
		vector will be printed.
		Otherwise, the pointer to Y is printed.
	*/
	void print();
};

#endif
