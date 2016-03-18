
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
	integrator.network.massTol = 1.0e-7; //not really necessary as sumX is always 1 by definition
	integrator.network.fluxFrac = 1e7; // disallows dtFlux, and thus dt from being larger than fluxFrac/maxFlux

	integrator.network.allocate();
	integrator.network.loadNetwork("AtmSpec_case0.inp");
	integrator.network.loadReactions("rateLibrary_atmosCHASER.data");
  integrator.network.loadPhotolytic("rateLibrary_atmosCHASER_photo.data");

	// Create the unique integration data

	{
		IntegrationData integrationData;
		integrationData.allocate(integrator.network.species);
		integrationData.loadAbundances("AtmSpec_case0.inp");
		
		integrationData.T = 290; //temp in Kelvin
    integrationData.M = 2.53e+19; // air number density (cm^-3)
    integrationData.H2O = 3.87E+17; //water vapor density (cm^-3)
    integrationData.Patm = 1.0; //atmospheric pressure in atm
    integrationData.pmb = 1013; //air pressure in millibar, for Y conversion from ppb to molecules/cm^3
    integrationData.zenith = 0.0; //0.0 indicates sun is overhead, pi/2 indicates sundown
    integrationData.alt = 2000; //up to 12000m
		integrationData.t_init = 1.0e-20;
		integrationData.t_max = 1.00e6;
		integrationData.dt_init = 3600;
		integrationData.rho = 1.0;

		// Launch the integrator
		integrator.integrate(integrationData);
		integrationData.print();
	}
	
	return EXIT_SUCCESS;
}
