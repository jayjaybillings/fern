#include <cstdlib>
#include "Globals.hpp"
#include "Network.hpp"

Globals::Globals(const std::shared_ptr<Network> & network)
{
	preFac = std::vector<fern_real>(network->reactions);
	Fplus = std::vector<fern_real>(network->totalFplus);
	Fminus = std::vector<fern_real>(network->totalFminus);
	rate = std::vector<fern_real>(network->reactions);
	massNum = std::vector<fern_real>(network->species);
	X =  std::vector<fern_real>(network->species);
	Fdiff = std::vector<fern_real>(network->species);
	Yzero = std::vector<fern_real>(network->species);
	FplusSum = std::vector<fern_real>(network->totalFplus);
	FminusSum = std::vector<fern_real>(network->totalFminus);
	Flux = std::vector<fern_real>(network->reactions);
}
