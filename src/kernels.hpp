
#ifndef kernels_cuh
#define kernels_cuh

#include "Network.hpp"
#include "IntegrationData.hpp"
#include "Globals.hpp"
#include <cstdlib>


void integrateNetwork(
	Network network,
	IntegrationData integrationData,
	Globals *globalsPtr
);


bool checkAsy(fern_real, fern_real, fern_real);
fern_real asymptoticUpdate(fern_real, fern_real, fern_real, fern_real);
fern_real eulerUpdate(fern_real, fern_real, fern_real, fern_real);

void populateF(fern_real *Fsign, fern_real *FsignFac, fern_real *Flux,
	unsigned short *MapFsign, unsigned short totalFsign);
inline void updatePopulations(fern_real *FplusSum, fern_real *FminusSum,
	fern_real *Y, fern_real *Yzero, unsigned short numberSpecies, fern_real dt);

fern_real *renormalize(fern_real *x, int n, fern_real sumX);
fern_real NDreduceSum(fern_real *a, unsigned short length);
fern_real reduceMax(fern_real *a, unsigned short length);
//EVENTUALLY INSERT PE function
void partialEquil(fern_real *Y, unsigned short numberReactions, int *ReacGroups, int **reactant, int **product, fern_real **final_k, int *pEquilbyRG, int *pEquilbyReac, int *ReacRG, int *RGid, int numRG, fern_real tolerance, int eq, fern_real *mostDevious, int *mostDeviousIndex);
void handlePERG_1(int i, fern_real y_a, fern_real y_b, fern_real *y_eq_a, fern_real kf, fern_real kr, int *pEquilbyRG, fern_real tolerance, fern_real *equilRatio);
void handlePERG_2(int i, fern_real y_a, fern_real y_b, fern_real y_c, fern_real *y_eq_a, fern_real kf, fern_real kr, int *pEquilbyRG, fern_real tolerance, fern_real *equilRatio);
void handlePERG_3(int i, fern_real y_a, fern_real y_b, fern_real y_c, fern_real y_d, fern_real *y_eq_a, fern_real kf, fern_real kr, int *pEquilbyRG, fern_real tolerance, fern_real *equilRatio);
void handlePERG_4(int i, fern_real y_a, fern_real y_b, fern_real y_c, fern_real y_d, fern_real *y_eq_a, fern_real kf, fern_real kr, int *pEquilbyRG, fern_real tolerance, fern_real *equilRatio);
void handlePERG_5(int i, fern_real y_a, fern_real y_b, fern_real y_c, fern_real y_d, fern_real y_e, fern_real *y_eq_a, fern_real kf, fern_real kr, int *pEquilbyRG, fern_real tolerance, fern_real *equilRatio);


void network_print(const Network &network);

#endif
