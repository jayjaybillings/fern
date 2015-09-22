#include <stdlib.h>
#include <stdio.h>
#include "FERNIntegrator.hpp"
#include <SimpleIni.h>

/**
 * This operation loads parameters from the input file into the integrator. It
 * will exit with a the EXIT_FAILURE status if the file cannot be loaded. It
 * calls the allocate() operation for both the integrator and the integration
 * data.
 * @param integrator the integrator into which the parameters should be loaded
 * @param filename the name of the file that contains the parameters
 */
void loadParameters(FERNIntegrator & integrator, IntegrationData & data,
		const char * filename) {

	// Load the parameters file
	CSimpleIniA iniReader;
	iniReader.SetUnicode();
	SI_Error status = iniReader.LoadFile(filename);
	// Exit with a failure if the file won't load.
	if (status < 0)
		exit(EXIT_FAILURE);

	// Load the network parameters. The simple parameters can be loaded
	// directly, but the file names are used in multiple places.
	integrator.network.species = atoi(
			iniReader.GetValue("network", "numSpecies", "0"));
	integrator.network.reactions = atoi(
			iniReader.GetValue("network", "numReactions", "0"));
	integrator.network.numRG = atoi(
			iniReader.GetValue("network", "numReactionGroups", "0"));
	integrator.network.massTol = strtod(
			iniReader.GetValue("network", "massTol", "0.0"), NULL);
	integrator.network.fluxFrac = strtod(
			iniReader.GetValue("network", "fluxFrac", "0.0"), NULL);
	const char * networkFile = iniReader.GetValue("network", "networkFile",
	NULL);
	const char * rateFile = iniReader.GetValue("network", "rateFile", NULL);

	// Load the solver parameters
	data.T9 = strtod(iniReader.GetValue("initialConditions", "T9",
			"0.0"), NULL);
	data.t_init = strtod(iniReader.GetValue("initialConditions",
			"startTime", "0.0"), NULL);
	data.t_max = strtod(iniReader.GetValue("initialConditions",
			"endTime", "0.0"), NULL);
	data.dt_init = strtod(iniReader.GetValue("initialConditions",
			"initialTimeStep", "0.0"), NULL);
	data.rho = strtod(iniReader.GetValue("initialConditions",
			"density", "0.0"), NULL);

	// Configure the integrator
	integrator.network.allocate();
	integrator.network.loadNetwork(networkFile);
	integrator.network.loadReactions(rateFile);
	// Configure the data
	data.allocate(integrator.network.species);
	data.loadAbundances(networkFile);

	return;
}

/**
 * This is the main function that starts the solve.
 * @param argc the number of input arguments
 * @param argv the array of input arguments
 * @return EXIT_SUCCESS (normally 0) if successful, EXIT_FAILURE otherwise.
 */
int main(int argc, char const *argv[]) {

	// There should only be two arguments
	if (argc != 2) {
		printf("Error! Invalid number of input arguments!\n");
		printf("Usage: ./fern-exec <input-file-path>\n");
		return EXIT_FAILURE;
	}

	/* Load the network */
	FERNIntegrator integrator;
	IntegrationData integrationData;
	loadParameters(integrator, integrationData, argv[1]);
	// Launch the integrator
	integrator.integrate(integrationData);
	integrationData.print();

	return EXIT_SUCCESS;
}
