
#include <stdlib.h>
#include <stdio.h>
#include "FERNIntegrator.cuh"

int main(int argc, char const *argv[])
{
	/* Load the network */
	FERNIntegrator integrator;
	integrator.network.species = 150;
	integrator.network.reactions = 1604;
	integrator.network.massTol = 1.0e-7;
	integrator.network.fluxFrac = 0.01;

	integrator.network.allocate();
	integrator.network.loadNetwork("CUDAnet_150.inp");
	integrator.network.loadReactions("rateLibrary_150.data");

	/* Initialize the solver */

	integrator.initializeCuda();
	integrator.prepareKernel();

	// Create the unique integration data

	{
		IntegrationData integrationData;
		integrationData.allocate(integrator.network.species);
		integrationData.loadAbundances("CUDAnet_150.inp");
		
		integrationData.T9 = 7.0;
		integrationData.t_init = 1.0e-20;
		integrationData.t_max = 1.0e-9;
		integrationData.dt_init = 1.23456789e-22;
		integrationData.rho = 1.0e8;

		// Launch the kernel
		
		integrator.integrate(integrationData);
		
		integrationData.print();
	}
	
	return EXIT_SUCCESS;
}
