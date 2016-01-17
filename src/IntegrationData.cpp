#include "IntegrationData.hpp"
#include "Network.hpp"
#include <cstdio>
#include <cstdlib>


void IntegrationData::loadAbundances(const char *filename)
{
	char ignoredData[16];
	fern_real ignoredDouble;
	int ignoredInt;
	
	FILE *file = fopen(filename, "r");
	
	// Exit if the file doesn't exist or can't be read
	
	if (!file)
	{
		fprintf(stderr, "Could not read file '%s'\n", filename);
		exit(1);
	}
	
	// Read 4 lines at a time
	
	for (int n = 0; n < species; n++)
	{
		int status;
		
		// Line #1
		
		#ifdef FERN_SINGLE
			status = fscanf(file, "%s %s %d %d %d %f %f\n",
				ignoredData, ignoredData, &ignoredInt, &ignoredInt,
				&ignoredInt, &Y[n], &ignoredDouble);
		#else
			status = fscanf(file, "%s %s %d %d %d %lf %lf\n",
				ignoredData, ignoredData, &ignoredInt, &ignoredInt,
				&ignoredInt, &Y[n], &ignoredDouble);
		#endif
		
		if (status == EOF)
			break;
		
		// Line #2...4
		
		for (int i = 0; i < 8 * 3; i++)
		{
			#ifdef FERN_SINGLE
				status = fscanf(file, "%f", &ignoredDouble);
			#else
				status = fscanf(file, "%lf", &ignoredDouble);
			#endif
		}
	}
}


void IntegrationData::allocate(unsigned short species)
{
	this->species = species;
	Y = new fern_real[species];
}

void IntegrationData::print()
{
	printf("species: %d\n", species);
	
	printf("T9: %e\n", T);
  printf("M: %e\n", M);
  printf("H2O: %e\n", H2O);
  printf("Patm: %e\n", Patm);
	printf("t_init: %e\n", t_init);
	printf("t_max: %e\n", t_max);
	printf("dt_init: %e\n", dt_init);
	printf("rho: %e\n", rho);
	
	printf("Y: ");
  fern_real Yppb = 0;
	for (unsigned short i = 0; i < species; i++){
    //convert back to ppb
    Yppb = (Y[i]*1e9)/(pmb*7.2428e+18/T); //Y[i] in ppb  
		//printf("%d: %e\n", i, Yppb);
		printf("%d: %e (ppb)\n", i, Yppb); //in molecules/cm^3
//		printf("%d: %e (cm)\n", i, Y[i]); //in molecules/cm^3
  }
}
