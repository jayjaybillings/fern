
#ifndef kernels_cuh
#define kernels_cuh

#include "Network.hpp"
#include "IntegrationData.hpp"
#include "Globals.hpp"


__global__ void integrateNetwork(
	Network network,
	IntegrationData integrationData,
	Globals *globalsPtr
);


bool checkAsy(fern_real, fern_real, fern_real);
fern_real asymptoticUpdate(fern_real, fern_real, fern_real, fern_real);
fern_real eulerUpdate(fern_real, fern_real, fern_real, fern_real);

vice__ void populateF(fern_real *Fsign, fern_real *FsignFac, fern_real *Flux,
	unsigned short *MapFsign, unsigned short totalFsign);
inline void updatePopulations(fern_real *FplusSum, fern_real *FminusSum,
	fern_real *Y, fern_real *Yzero, unsigned short numberSpecies, fern_real dt);

fern_real reduceSum(fern_real *a, unsigned short length);
fern_real NDreduceSum(fern_real *a, unsigned short length);
fern_real reduceMax(fern_real *a, unsigned short length);


size_t integrateNetwork_sharedSize(const Network &network);
void network_print(const Network &network);

#endif
