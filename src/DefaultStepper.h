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
#ifndef DEFAULTSTEPPER_H_
#define DEFAULTSTEPPER_H_

#include <algorithm>
#include <Globals.hpp>
#include <Network.hpp>
#include <fern_math.h>
#include <iostream>

/**
 * This class is the default time stepper for FERN.
 */
class DefaultStepper: public fire::IStepper {

private:

	/// Initial time
	double t_init = 0.0;
	/// Final time
	double t_max = 0.0;
	/// Current time
	double t;
	/// Current time step
	double dt = 0.0;
	/// Previous mass fraction sum
	double sumXLast = 0.0;
	/// Total number of time steps
	int timesteps = 0;
	/// Restart time marker
	double deltaTimeRestart = 0.0;
	/// The initial step size
	double initialStepSize = 0.0;
	/// Sum of the mass fractions
	fern_real sumX;
	/// Abundance values
	const fern_real * Y;
	/// Global variables
	const Globals & globals;
	/// Network of reactions and reactants
	const Network & network;

public:

	/**
	 * The constructor
	 * @param globalsRef Reference to the global data array
	 * @param networkRef Reference to the network information
	 * @param yArray Pointer to abundance values
	 */
	DefaultStepper(const Globals & globalsRef, const Network & networkRef,
			const fern_real * yArray) :
			globals(globalsRef), network(networkRef), Y(yArray) {
	}
	;

	virtual ~DefaultStepper() {
	}
	;

	double getStep() {
		return t;
	}
	;

	double getStepSizeAtStage(int i) {
		double size = 0.0;
		if (i == 1) {

			fern_real floorFac = 0.1;

			// Get the max flux for the time step calculation
			fern_real maxFlux = *(max_element(globals.Fdiff.begin(),
					globals.Fdiff.end()));

			/*
			 Now use the fluxes to update the populations for this timestep.
			 For now we shall assume the asymptotic method. We determine whether each isotope
			 satisfies the asymptotic condition. If it does we update with the asymptotic formula.
			 If not, we update numerically using the forward Euler formula.
			 */

			/* Determine an initial trial timestep based on fluxes and dt in previous step. */
			fern_real dtFlux = network.fluxFrac / maxFlux;
			fern_real dtFloor = floorFac * t;
			dtFlux = std::min(dtFlux,dtFloor);

			// The time step for this stage is the minimum of the time step
			// computed based on flux considerations and the restart time.
			dt = std::min(deltaTimeRestart, dtFlux);
	//		std::cout << "t1 = " << dt << std::endl;
		} else if (i == 2) {

			fern_real upbumper = 0.9 * network.massTol;
			fern_real downbumper = 0.1;
			fern_real massTolUp = 0.25 * network.massTol;

			// Sum the mass fractions
			sumX = 0.0;
			for (double x : globals.X) {
				sumX += x;
			}

			/*
			 Now modify timestep if necessary to ensure that particle number is conserved to
			 specified tolerance (but not too high a tolerance). Using updated populations
			 based on the trial timestep computed above, test for conservation of particle
			 number and modify trial timestep accordingly.
			 */
			fern_real test1 = sumXLast - 1.0;
			fern_real test2 = sumX - 1.0;
			fern_real massChecker = std::abs(sumXLast - sumX);

			if (std::abs(test2) > std::abs(test1) && massChecker > network.massTol) {
				dt *= fmax( network.massTol	/ massChecker, downbumper);
			} else if (massChecker < massTolUp) {
				dt *= (network.massTol / (fmax(massChecker, upbumper)));
			}

			/*
			 Store the actual timestep that would be taken. Same as dt unless
			 artificially shortened in the last integration step to match end time.
			 */
			deltaTimeRestart = dt;
	//		std::cout << "t2 = " << dt << std::endl;
		}

		// Adjust the time if crosses over the maximum time.
		if (t + dt >= t_max) {
			dt = t_max - t;
		}

		return dt;
	}
	;

	void updateStep() {
		t += dt;
		++timesteps;
		// Update sum of the mass fractions
		sumX = 0.0;
		for (double x : globals.X) {
			sumX += x;
		}
		sumXLast = sumX;
		// FIXME! This does not accurately account of PE!
	}
	;

	/**
	 * This operation sets the initial step size for the stepper
	 * @param the initial step size
	 */
	void setInitialStepsize(double stepSize) {
		initialStepSize = stepSize;
		deltaTimeRestart = stepSize;
	}

	/**
	 * This operation gets the initial step size for the stepper
	 * @return the initial step size
	 */
	double getInitialStepsize() {
		return initialStepSize;
	}

	double getInitialStep() {
		// Compute the initial sum of the mass fractions. This has to be done
		// here because it affects the tests for the time stepping.
		for (double x : globals.X) {
			sumXLast += x;
		}
		return t_init;
	}
	;

	/**
	 * This operation sets the initial step for the stepper.
	 * @param initialStep the initial step
	 */
	void setInitialStep(double initialStep) {
		t_init = initialStep;
		t = t_init;
	}

	double getFinalStep() {
		return t_max;
	}

	/**
	 * This operation sets the final step for the stepper.
	 * @param finalStep the final step
	 */
	void setFinalStep(double finalStep) {
		t_max = finalStep;
	}
};

#endif /* DEFAULTSTEPPER_H_ */
