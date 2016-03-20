
#include <stdlib.h>
#include <stdio.h>
#include "FERNIntegrator.hpp"

int main(int argc, char const *argv[])
{
	/* Load the network */
  bool whichNetwork = 0; // 0 is 150-isotope, 1 is alpha-network
	FERNIntegrator integrator;
  if (whichNetwork == 0) {
	  integrator.network.species = 150;
  	integrator.network.reactions = 1604;
    integrator.network.numRG = 741;
  } else {
	  integrator.network.species = 16;
    integrator.network.reactions = 48;
    integrator.network.numRG = 19;
  }
	integrator.network.massTol = 1;
	integrator.network.fluxFrac = .2;
	integrator.network.allocate();

  if (whichNetwork == 0) {
	  integrator.network.loadNetwork("CUDAnet_150.inp");
  	integrator.network.loadReactions("rateLibrary_150.data");
  } else {
  	integrator.network.loadNetwork("CUDAnet_alpha.inp");
	  integrator.network.loadReactions("rateLibrary_alpha.data");
  }

	// Create the unique integration data

	{
		IntegrationData integrationData;
		integrationData.allocate(integrator.network.species);

    if (whichNetwork == 0) {
  		integrationData.loadAbundances("CUDAnet_150.inp");
		  integrationData.t_max = 1.0e-8;
    } else {
		  integrationData.loadAbundances("CUDAnet_alpha.inp");
		  integrationData.t_max = 1.0e-2;
    }
		
		integrationData.T9 = 7.0;
		integrationData.t_init = 1.0e-20;
		integrationData.dt_init = 1.23456789e-22;
		integrationData.rho = 1.0e8;

		// Launch the integrator
		integrator.integrate(integrationData);
		integrationData.print();
	}
	
	return EXIT_SUCCESS;
}
