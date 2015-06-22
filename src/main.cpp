
#include <stdlib.h>
#include <stdio.h>
#include "FERNIntegrator.hpp"

int main(int argc, char const *argv[])
{
	/* Load the network */
	FERNIntegrator integrator;
	integrator.network.species = 102;
	integrator.network.reactions = 114;
	integrator.network.photoparams = 40;
	integrator.network.photolytic = 26;
	integrator.network.numRG = 101;
	integrator.network.massTol = 1.0e-7;
	integrator.network.fluxFrac = 0.01;

	integrator.network.allocate();
	integrator.network.loadNetwork("AtmSpec_polluted.inp");
	integrator.network.loadReactions("rateLibrary_atmosCHASER.data");
  integrator.network.loadPhotolytic("rateLibrary_atmosCHASER_photo.data");

	// Create the unique integration data

	{
		IntegrationData integrationData;
		integrationData.allocate(integrator.network.species);
		integrationData.loadAbundances("AtmSpec_polluted.inp");
		
		integrationData.T9 = 290;
    integrationData.H2O = 1.0;
    integrationData.M = 1.0;
    integrationData.Patm = 1.0;
    integrationData.zenith = 0.0; //0.0 indicates sun is overhead
    integrationData.alt = 1000; //up to 12000m
		integrationData.t_init = 1.0e-20;
		integrationData.t_max = 1.0e-5;
		integrationData.dt_init = 1.23456789e-22;
		integrationData.rho = 1.0;

		// Launch the integrator
		integrator.integrate(integrationData);
		integrationData.print();
	}
	
	return EXIT_SUCCESS;
}
