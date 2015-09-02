#include "Globals.cuh"
#include "Network.cuh"


void Globals::cudaAllocate(const Network &network)
{
	cudaMalloc(&preFac, sizeof(fern_real) * network.reactions);
	cudaMalloc(&Fplus, sizeof(fern_real) * network.totalFplus);
	cudaMalloc(&Fminus, sizeof(fern_real) * network.totalFminus);
	cudaMalloc(&rate, sizeof(fern_real) * network.reactions);
	cudaMalloc(&massNum, sizeof(fern_real) * network.species);
	cudaMalloc(&X, sizeof(fern_real) * network.species);
	cudaMalloc(&Fdiff, sizeof(fern_real) * network.species);
	cudaMalloc(&Yzero, sizeof(fern_real) * network.species);
	cudaMalloc(&FplusSum, sizeof(fern_real) * network.totalFplus);
	cudaMalloc(&FminusSum, sizeof(fern_real) * network.totalFminus);
	cudaMalloc(&Flux, sizeof(fern_real) * network.reactions);
}
