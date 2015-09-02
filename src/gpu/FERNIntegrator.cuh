
// #define FERN_SINGLE	1

#ifdef FERN_SINGLE
	typedef float fern_real;
#else
	typedef double fern_real;
#endif

#ifndef FERNIntegrator_cuh
#define FERNIntegrator_cuh

#include "IntegrationData.cuh"


class FERNIntegrator
{
public:
	FERNIntegrator();
	~FERNIntegrator();
	
	// void loadProperties(const char *filename);
	
	void initializeCuda();
	void prepareKernel();
	
	/**	Launches a kernel to process the IntegrationData
	*/
	void integrate(IntegrationData &integrationData);
	static void checkCudaErrors();
	
	Network network;
	
private:
	Network devNetwork;
	
	dim3 blocks;
	dim3 threads;
	
	static void devcheck(int gpudevice);
};

#endif
