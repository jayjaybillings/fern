
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
	integrator.network.fluxFrac = 1e8;

	integrator.network.allocate();
	integrator.network.loadNetwork("newAtmSpec_polluted.inp");
	integrator.network.loadReactions("rateLibrary_atmosCHASER.data");
  integrator.network.loadPhotolytic("rateLibrary_atmosCHASER_photo.data");

	// Create the unique integration data

	{
		IntegrationData integrationData;
		integrationData.allocate(integrator.network.species);
		integrationData.loadAbundances("newAtmSpec_polluted.inp");
		
		integrationData.T = 290; //temp in Kelvin
    integrationData.H2O = 1.0; //water vapor density (cm^-3)
    integrationData.M = 1.0; // air number density (cm^-3)
    integrationData.Patm = 1.0; //atmospheric pressure in atm
    integrationData.pmb = 1000; //air pressure in millibar, for Y conversion from ppb to molecules/cm^3
    integrationData.zenith = 0.0; //0.0 indicates sun is overhead, pi/2 indicates sundown
    integrationData.alt = 2000; //up to 12000m
		integrationData.t_init = 1.0e-20;
		integrationData.t_max = 3.0e+2;
		integrationData.dt_init = 1e-10;
		integrationData.rho = 1.0;

		// Launch the integrator
		integrator.integrate(integrationData);
		integrationData.print();
	}
	
	return EXIT_SUCCESS;
}
