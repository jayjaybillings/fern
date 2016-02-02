
#include <stdlib.h>
#include <stdio.h>
#include "FERNIntegrator.hpp"

int main(int argc, char const *argv[])
{
	/* Load the network */
	FERNIntegrator integrator;
	//integrator.network.species = 150;
	integrator.network.species = 16;
	//integrator.network.reactions = 1604;
	integrator.network.reactions = 48;
	integrator.network.massTol = 1;
	integrator.network.fluxFrac = .000001;
  //integrator.network.numRG = 741;
  integrator.network.numRG = 19;

	integrator.network.allocate();
	//integrator.network.loadNetwork("CUDAnet_150.inp");
	integrator.network.loadNetwork("CUDAnet_alpha.inp");
	//integrator.network.loadReactions("rateLibrary_150.data");
	integrator.network.loadReactions("rateLibrary_alpha.data");

	// Create the unique integration data

	{
		IntegrationData integrationData;
		integrationData.allocate(integrator.network.species);
		integrationData.loadAbundances("CUDAnet_alpha.inp");
		
		integrationData.T9 = 7.0;
		integrationData.t_init = 1.0e-20;
		integrationData.t_max = 1.0e-2;
		integrationData.dt_init = 1.23456789e-22;
		integrationData.rho = 1.0e8;

		// Launch the integrator
		integrator.integrate(integrationData);
		integrationData.print();
	}
	
	return EXIT_SUCCESS;
}
