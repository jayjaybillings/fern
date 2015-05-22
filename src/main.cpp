
#include <stdlib.h>
#include <stdio.h>
#include "FERNIntegrator.hpp"

int main(int argc, char const *argv[])
{
	/* Load the network */
	FERNIntegrator integrator;
	integrator.network.species = 102;
	integrator.network.reactions = 112;
	integrator.network.massTol = 1.0e-7;
	integrator.network.fluxFrac = 0.01;

	integrator.network.allocate();
	integrator.network.loadNetwork("CUDAnet_atmos_kengo.inp");
	integrator.network.loadReactions("rateLibrary_atmos.data");

	// Create the unique integration data

	{
		IntegrationData integrationData;
		integrationData.allocate(integrator.network.species);
		integrationData.loadAbundances("CUDAnet_atmos_kengo.inp");
		
		integrationData.T9 = 7.0;
		integrationData.t_init = 1.0e-20;
		integrationData.t_max = 1.0e-5;
		integrationData.dt_init = 1.23456789e-22;
		integrationData.rho = 1.0e8;

		// Launch the integrator
		integrator.integrate(integrationData);
		integrationData.print();
	}
	
	return EXIT_SUCCESS;
}
