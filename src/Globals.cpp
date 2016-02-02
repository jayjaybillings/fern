#include <cstdlib>
#include "Globals.hpp"
#include "Network.hpp"

void Globals::allocate(Network &network)
{
	preFac = (fern_real *) malloc(sizeof(fern_real) * network.reactions);
	Fplus = (fern_real *) malloc(sizeof(fern_real) * network.totalFplus);
	Fminus = (fern_real *) malloc(sizeof(fern_real) * network.totalFminus);
	FplusBefore = (fern_real *) malloc(sizeof(fern_real) * network.totalFplus);
	FminusBefore = (fern_real *) malloc(sizeof(fern_real) * network.totalFminus);
	rate = (fern_real *) malloc(sizeof(fern_real) * network.reactions);
	massNum = (fern_real *) malloc(sizeof(fern_real) * network.species);
	X = (fern_real *) malloc(sizeof(fern_real) * network.species);
	Fdiff = (fern_real *) malloc(sizeof(fern_real) * network.species);
	Yzero = (fern_real *) malloc(sizeof(fern_real) * network.species);
	FplusSum = (fern_real *) malloc(sizeof(fern_real) * network.totalFplus);
	FminusSum = (fern_real *) malloc(sizeof(fern_real) * network.totalFminus);
	FplusSumBefore = (fern_real *) malloc(sizeof(fern_real) * network.totalFplus);
	FminusSumBefore = (fern_real *) malloc(sizeof(fern_real) * network.totalFminus);
	Flux = (fern_real *) malloc(sizeof(fern_real) * network.reactions);
}
