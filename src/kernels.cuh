
#ifndef kernels_cuh
#define kernels_cuh

#include "Network.cuh"
#include "IntegrationData.cuh"
#include "Globals.cuh"


__global__ void integrateNetwork(
	Network network,
	IntegrationData integrationData,
	Globals *globalsPtr
);


__device__ bool checkAsy(fern_real, fern_real, fern_real);
__device__ fern_real asymptoticUpdate(fern_real, fern_real, fern_real, fern_real);
__device__ fern_real eulerUpdate(fern_real, fern_real, fern_real, fern_real);

__device__ void populateF(fern_real *Fsign, fern_real *FsignFac, fern_real *Flux,
	unsigned short *MapFsign, unsigned short totalFsign);
__device__ inline void updatePopulations(fern_real *FplusSum, fern_real *FminusSum,
	fern_real *Y, fern_real *Yzero, unsigned short numberSpecies, fern_real dt);

__device__ fern_real reduceSum(fern_real *a, unsigned short length);
__device__ fern_real NDreduceSum(fern_real *a, unsigned short length);
__device__ fern_real reduceMax(fern_real *a, unsigned short length);


size_t integrateNetwork_sharedSize(const Network &network);
__device__ void network_print(const Network &network);

#endif
