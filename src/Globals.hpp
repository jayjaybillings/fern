#include "FERNIntegrator.hpp"

#ifndef Globals_cuh
#define Globals_cuh

struct Network;

/**	Only allocated on the device per FERN integration
*/
struct Globals
{
	fern_real *preFac; // [reactions]
	fern_real *Flux; // [reactions]
	fern_real *Fplus; // [totalFplus]
	fern_real *Fminus; // [totalFminus]
	fern_real *FplusBefore; // [totalFplus]
	fern_real *FminusBefore; // [totalFminus]
	fern_real *rate; // [reactions]
	fern_real *massNum;
	fern_real *X;
	fern_real *Fdiff;
	fern_real *Yzero;
	fern_real *FplusSum;
	fern_real *FminusSum;
	fern_real *FplusSumBefore;
	fern_real *FminusSumBefore;

	void allocate(Network &network);
};

#endif
