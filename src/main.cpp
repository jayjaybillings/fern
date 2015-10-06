/**----------------------------------------------------------------------------
Copyright (c) 2015-, The University of Tennessee
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of fern nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author(s): Jay Jay Billings, Ben Brock, Andrew Belt, Dan Shyles, Mike Guidry
-----------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include "kernels.hpp"
#include <SimpleIni.h>

/**
 * This operation loads parameters from the input file into the integrator. It
 * will exit with a the EXIT_FAILURE status if the file cannot be loaded. It
 * calls the allocate() operation for both the integrator and the integration
 * data.
 * @param integrator the integrator into which the parameters should be loaded
 * @param filename the name of the file that contains the parameters
 */
void loadParameters(Network * network, IntegrationData * data,
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
	network->species = atoi(
			iniReader.GetValue("network", "numSpecies", "0"));
	network->reactions = atoi(
			iniReader.GetValue("network", "numReactions", "0"));
	network->numRG = atoi(
			iniReader.GetValue("network", "numReactionGroups", "0"));
	network->massTol = strtod(
			iniReader.GetValue("network", "massTol", "0.0"), NULL);
	network->fluxFrac = strtod(
			iniReader.GetValue("network", "fluxFrac", "0.0"), NULL);
	const char * networkFile = iniReader.GetValue("network", "networkFile",
	NULL);
	const char * rateFile = iniReader.GetValue("network", "rateFile", NULL);

	// Load the solver parameters
	data->T9 = strtod(iniReader.GetValue("initialConditions", "T9",
			"0.0"), NULL);
	data->t_init = strtod(iniReader.GetValue("initialConditions",
			"startTime", "0.0"), NULL);
	data->t_max = strtod(iniReader.GetValue("initialConditions",
			"endTime", "0.0"), NULL);
	data->dt_init = strtod(iniReader.GetValue("initialConditions",
			"initialTimeStep", "0.0"), NULL);
	data->rho = strtod(iniReader.GetValue("initialConditions",
			"density", "0.0"), NULL);

	// Configure the integrator
	network->allocate();
	network->loadNetwork(networkFile);
	network->loadReactions(rateFile);
	// Configure the data
	data->allocate(network->species);
	data->loadAbundances(networkFile);

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
	IntegrationData integrationData;
	Network network;
	loadParameters(&network, &integrationData, argv[1]);
	// Launch the integrator
	Globals globals;
	globals.allocate(network);
	initialize(&network, &integrationData, &globals);
	integrate();
	integrationData.print();

	return EXIT_SUCCESS;
}
