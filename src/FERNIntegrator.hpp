
// #define FERN_SINGLE	1

#ifdef FERN_SINGLE
	typedef float fern_real;
#else
	typedef double fern_real;
#endif

#ifndef FERNIntegrator_cuh
#define FERNIntegrator_cuh

#include "IntegrationData.hpp"


class FERNIntegrator
{
public:
	FERNIntegrator();
	~FERNIntegrator();
	
	/**	Launches a kernel to process the IntegrationData
	*/
	void integrate(IntegrationData &integrationData);
	
	Network network;
};

#endif
