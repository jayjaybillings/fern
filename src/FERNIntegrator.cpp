
#include "FERNIntegrator.hpp"
#include "Globals.hpp"
#include "kernels.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>


FERNIntegrator::FERNIntegrator()
{
	// TODO
	// Hardcoded for now. Make this the default for a command line
	// argument or something.
}


FERNIntegrator::~FERNIntegrator()
{
	// TODO
	// Implement the freeing methods
	
	// devReactionData.cudaFree();
	// devNetworkData.cudaFree();
	
	// reactionData.free();
	// networkData.free();
}

void FERNIntegrator::integrate(IntegrationData &integrationData)
{
	Globals globals;
	globals.allocate(network);

	integrateNetwork(network, integrationData, &globals);
}
