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
#ifndef kernels_cuh
#define kernels_cuh

#include "Network.hpp"
#include "IntegrationData.hpp"
#include "Globals.hpp"
#include "fern_math.h"
#include <memory>

/**
 * This operation initializes the solver.
 * @param network the network that should be integrated
 * @param integrationData the set of integration parameters for the integration
 * @param globals the set of global variables
 */
void initialize(std::shared_ptr<Network> network,
		IntegrationData * integrationData, std::shared_ptr<Globals> globalsPtr);

/**
 * This operation performs the integration based on the information provided to
 * the initialization routine.
 */
void integrate();

bool checkAsy(fern_real, fern_real, fern_real);
fern_real asymptoticUpdate(fern_real, fern_real, fern_real, fern_real);
fern_real eulerUpdate(fern_real, fern_real, fern_real, fern_real);

void populateF(std::vector<fern_real> Fsign,
		const std::vector<fern_real> & FsignFac, std::vector<fern_real> Flux,
		const std::vector<unsigned short> & MapFsign,
		unsigned short totalFsign);
inline void updatePopulations(std::vector<fern_real> FplusSum,
		std::vector<fern_real> FminusSum, fern_real *Y, std::vector<fern_real> Yzero,
		unsigned short numberSpecies, fern_real dt);

fern_real NDreduceSum(std::vector<fern_real> a, unsigned short length);
fern_real reduceMax(std::vector<fern_real> a, unsigned short length);
//EVENTUALLY INSERT PE function
void partialEquil(fern_real *Y, unsigned short numberReactions, int *ReacGroups,
		int **reactant, int **product, fern_real **final_k, int *pEquilbyRG,
		int *pEquilbyReac, int *ReacRG, int *RGid, int numRG,
		fern_real tolerance, int eq);
void handlePERG_1(int i, fern_real y_a, fern_real y_b, fern_real kf,
		fern_real kr, int *pEquilbyRG, fern_real tolerance);
void handlePERG_2(int i, fern_real y_a, fern_real y_b, fern_real y_c,
		fern_real kf, fern_real kr, int *pEquilbyRG, fern_real tolerance);
void handlePERG_3(int i, fern_real y_a, fern_real y_b, fern_real y_c,
		fern_real y_d, fern_real kf, fern_real kr, int *pEquilbyRG,
		fern_real tolerance);
void handlePERG_4(int i, fern_real y_a, fern_real y_b, fern_real y_c,
		fern_real y_d, fern_real kf, fern_real kr, int *pEquilbyRG,
		fern_real tolerance);
void handlePERG_5(int i, fern_real y_a, fern_real y_b, fern_real y_c,
		fern_real y_d, fern_real y_e, fern_real kf, fern_real kr,
		int *pEquilbyRG, fern_real tolerance);

void network_print(const Network &network);

#endif
