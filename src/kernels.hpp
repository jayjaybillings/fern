
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


bool checkAsy(fern_real, fern_real, fern_real);
fern_real asymptoticUpdate(fern_real, fern_real, fern_real, fern_real);
fern_real eulerUpdate(fern_real, fern_real, fern_real, fern_real);

void populateF(fern_real *Fsign, fern_real *FsignFac, fern_real *Flux,
	unsigned short *MapFsign, unsigned short totalFsign);
inline void updatePopulations(fern_real *FplusSum, fern_real *FminusSum,
	fern_real *Y, fern_real *Yzero, unsigned short numberSpecies, fern_real dt);

fern_real NDreduceSum(fern_real *a, unsigned short length);
fern_real reduceMax(fern_real *a, unsigned short length);
void partialEquil(fern_real *Y, unsigned short numberReactions, int *ReacGroups, unsigned short **reactant, unsigned short **product, fern_real **final_k, int *pEquil, int *RGid, int numRG, fern_real tolerance, int eq);


void network_print(const Network &network);

#endif
