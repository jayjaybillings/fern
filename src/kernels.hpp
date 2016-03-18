
#ifndef kernels_cuh
#define kernels_cuh

#include "Network.hpp"
#include "IntegrationData.hpp"
#include "Globals.hpp"
#include <cstdlib>


void integrateNetwork(
	Network network,
	IntegrationData integrationData,
	Globals *globalsPtr
);


fern_real SolarZenithAngle(fern_real td, fern_real latitude, fern_real longitude);
void calculatePhotolyticRates(int i, fern_real zenith, fern_real alt, int **paramNumID, fern_real **aparam, fern_real **paramMult, fern_real *Rate);
bool checkAsy(fern_real, fern_real, fern_real);
fern_real asymptoticUpdate(fern_real, fern_real, fern_real, fern_real);
fern_real eulerUpdate(fern_real, fern_real, fern_real, fern_real);

void populateF(fern_real *Fsign, fern_real *FsignFac, fern_real *Flux,
	unsigned short *MapFsign, unsigned short totalFsign);
inline void updatePopulations(fern_real *FplusSum, fern_real *FminusSum,
	fern_real *Y, fern_real *Yzero, unsigned short numberSpecies, fern_real dt);

fern_real NDreduceSum(fern_real *a, unsigned short length);
fern_real reduceMax(fern_real *a, unsigned short length);
void partialEquil(fern_real *Y, unsigned short numberReactions, int *ReacGroups, int **reactant, int **product, fern_real **final_k, int *pEquil, int *RGid, int numRG, fern_real tolerance, int eq);


void network_print(const Network &network);

#endif
