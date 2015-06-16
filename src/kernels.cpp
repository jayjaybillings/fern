#include <stdio.h>
#include <cmath>
#include "kernels.hpp"

void integrateNetwork(
	Network network,
	IntegrationData integrationData,
	Globals *globalsPtr
)
{
	Globals &globals = *globalsPtr;

	/*
	   TEMPORARY
	   Variables are declared with local pointers.
	   This is to ease refactoring and allow easy
	   maneuvering with dynamic shared memory.
	*/

	/* Declare local pointers for Globals arrays. */

	fern_real *Flux;
	fern_real *Fplus;
	fern_real *Fminus;
	fern_real *Rate;
	fern_real *massNum;
	fern_real *X;
	fern_real *Fdiff;
	fern_real *Yzero;
	fern_real *FplusSum;
	fern_real *FminusSum;
	
	/* Declare local variables for Network struct. */

	const unsigned short numberSpecies = network.species;
	const unsigned short numberReactions = network.reactions;
	const unsigned short totalFplus = network.totalFplus;
	const unsigned short totalFminus = network.totalFminus;
  

	unsigned char *Z;
	unsigned char *N;

	fern_real *FplusFac;
	fern_real *FminusFac;

	unsigned short *MapFplus;
	unsigned short *MapFminus;

	unsigned short *FplusMax;
	unsigned short *FminusMax;

	const fern_real massTol = network.massTol;
	const fern_real fluxFrac = network.fluxFrac;

	/* Declare pointer variables for IntegrationData arrays.  */

	fern_real *Y;

	const bool plotOutput = 1;
	const int numIntervals = 100;
	int plotStartTime = -16;
	fern_real intervalLogt;
  fern_real nextOutput;
  int outputCount = 0;
	int setNextOut = 0;
	fern_real asyCount = 0;
	fern_real peCount = 0;
  fern_real FracAsy = 0; 
  fern_real FracRGPE = 0;
	int eq = 0;
  int numRG = network.numRG;
  int *RGid;
  RGid = network.RGid;
  int *ReacParent;
  ReacParent = network.ReacParent;
  int *RGmemberIndex;
  RGmemberIndex = network.RGmemberIndex;
  int *isReverseR;
  isReverseR = network.isReverseR;
  int *ReacGroups;
  ReacGroups = network.ReacGroups;
  int *pEquil;
  pEquil = network.pEquil;

	/* Assign globals pointers. */
	
	Flux = globals.Flux;
	Fplus = globals.Fplus;
	Fminus = globals.Fminus;
	Rate = globals.rate;
	massNum = globals.massNum;
	X = globals.X;
	Fdiff = globals.Fdiff;
	Yzero = globals.Yzero;
	FplusSum = globals.FplusSum;
	FminusSum = globals.FminusSum;

	/* Assign Network pointers. */

	Z = network.Z;
	N = network.N;
	FplusFac = network.FplusFac;
	FminusFac = network.FminusFac;
	MapFplus = network.MapFplus;
	MapFminus = network.MapFminus;
	FplusMax = network.FplusMax;
	FminusMax = network.FminusMax;

	/* Assign IntegrationData pointers. */
	
	Y = integrationData.Y;

	fern_real maxFlux;
	fern_real sumX;
	fern_real t;
	fern_real dt;
	unsigned int timesteps;
	
	fern_real sumXLast;

	/* Compute the preFac vector. */
	
	for (int i = 0; i < network.reactions; i++)
	{
		#ifdef FERN_SINGLE
			globals.preFac[i] = network.statFac[i] *
				powf(integrationData.rho, network.numReactingSpecies[i] - 1);
		#else
			globals.preFac[i] = network.statFac[i] *
				pow(integrationData.rho, network.numReactingSpecies[i] - 1);
		#endif
	}

	/* Compute the rate values. */

	/*
	   Compute the temperature-dependent factors for the rates.
	   Since we assume the GPU integration to be done at constant
	   temperature and density, these only need be calculated once
	   per GPU call.
	*/

	fern_real T93 = cbrt(integrationData.T9);
	fern_real t1 = 1 / integrationData.T9;
	fern_real t2 = 1 / T93;
	fern_real t3 = T93;
	fern_real t4 = integrationData.T9;
	fern_real t5 = T93 * T93 * T93 * T93 * T93;
	fern_real t6 = log(integrationData.T9);

	for (int i = 0; i < network.reactions; i++)
	{
		#ifdef FERN_SINGLE
			Rate[i] = globals.preFac[i] * expf(
				     network.P[0][i] + t1 * network.P[1][i] +
				t2 * network.P[2][i] + t3 * network.P[3][i] +
				t4 * network.P[4][i] + t5 * network.P[5][i] +
				t6 * network.P[6][i]);
		#else
 			Rate[i] = globals.preFac[i] * exp(
				     network.P[0][i] + t1 * network.P[1][i] +
				t2 * network.P[2][i] + t3 * network.P[3][i] +
				t4 * network.P[4][i] + t5 * network.P[5][i] +
				t6 * network.P[6][i]);
		#endif
	}

	/* Author: Daniel Shyles */
	/* Begin Partial Equilibrium calculation */
	const bool displayRGdata = false;
    fern_real kf;
    fern_real kr;
    fern_real *final_k[2];
	int countRG = 0;
		//first set up array of final reaction rates for each RG based on Rate[i] calculated above
		for(int m = 0; m < 2; m++) {
			final_k[m] = new fern_real [network.numRG];
    }

		if(displayRGdata) {
			printf("Start Reaction Group Data\nNumber Reaction Groups: %d\n\n",network.numRG);
    }

		//second to calculate final reaciton rates for each RG
		for(int i = 0; i < network.reactions; i++) {
			if(displayRGdata && ReacGroups[i] != 0) {
				printf("RG #%d\nRG Class: %d\nParent Reaction: %d\n", countRG, ReacGroups[i], i);
				//output numReactingSpecies and numProducts for Parent of RG
        printf("numReacting: %d, numProducts: %d\n", network.numReactingSpecies[i], network.PEnumProducts[i]);			
				if(ReacGroups[i] == 1) {		
					printf("Reactant SID: %d; Product SID: %d\n",network.reactant[0][i], network.product[0][i]);
				} 
				else if(ReacGroups[i] == 2) {
          printf("Reactant SID: %d, %d; Product SID: %d\n",network.reactant[0][i], network.reactant[1][i], network.product[0][i]);
        }
        else if(ReacGroups[i] == 3) {
          printf("Reactant SID: %d, %d, %d; Product SID: %d\n",network.reactant[0][i], network.reactant[1][i], network.reactant[2][i], network.product[0][i]);
        }
        else if(ReacGroups[i] == 4) {
          printf("Reactant SID: %d, %d; Product SID: %d, %d\n",network.reactant[0][i], network.reactant[1][i], network.product[0][i], network.product[1][i]);
        }
        else if(ReacGroups[i] == 5) {
          printf("Reactant SID: %d, %d; Product SID: %d, %d, %d\n",network.reactant[0][i], network.reactant[1][i], network.product[0][i], network.product[1][i], network.product[2][i]);
        }
				printf("-----\n|\n");
			}				
			if(displayRGdata)
				printf("Reaction ID: %d\nRG Member ID: %d\nisReverseR: %d\nRate:%e\n|\n", i, RGmemberIndex[i], isReverseR[i], Rate[i]);

            //if RGmemberindex is greater (or equal for RGmemberindex[i] = RGmemberindex[i+1] = 0 than next one, then end of Reaction Group
            if(RGmemberIndex[i] >= RGmemberIndex[i+1]) {
                //get forward and reverse rates for all reactions within group, starting with i-network.RGmemberIndex[i], and ending with i.
                kf = 0; //forward rate
                kr = 0; //reverse rate
                //iterate through each RGmember and calculate the total rate from forward and reverse reactions
                for(int n = RGmemberIndex[i]; n >= 0; n--) {
                    //add the rate to forward reaction
                    if(isReverseR[i-n] == 1) {
                        kr += Rate[i-n];
                        //printf("Reverse Rate for RG#%d: %e, Sum: %e\n", countRG, Rate[i-n], kr);
                    } else {
                    //add the rate to reverse reaction
                        kf += Rate[i-n];
                    }
                }
                final_k[0][countRG] = 0;
                final_k[1][countRG] = 0;
                final_k[0][countRG] = kf;
                final_k[1][countRG] = kr;
                if(displayRGdata) {
                    printf("-----\n");
                    printf("kf[RGID:%d] = %e \n", countRG, final_k[0][countRG]);
                    printf("kr[RGID:%d] = %e \n", countRG, final_k[1][countRG]);
                    printf("\n\n\n");
                }
				    countRG++;
            }
		}
	/*
	   Begin the time integration from t=0 to tmax. Rather than t=0 we
	   start at some very small value of t. This is required for the CUDA C
	   code as well as the Java version.
	*/
	
	t = 1.0e-20;
	dt = integrationData.dt_init;
	timesteps = 1;
	
	fern_real floorFac = 0.1;
	fern_real upbumper = 0.9 * massTol;
	fern_real downbumper = 0.1;
	fern_real massTolUp = 0.25 * massTol;
	fern_real deltaTimeRestart = dt;
	fern_real dtFloor;
	fern_real dtFlux;
	fern_real massChecker;
	
	/* Compute mass numbers and initial mass fractions X for all isotopes. */
	
	for (int i = 0; i < numberSpecies; i++)
	{
		massNum[i] = (fern_real) Z[i] + (fern_real) N[i];
		/* Compute mass fraction X from abundance Y. */
		X[i] = massNum[i] * Y[i];
	}
	
	sumXLast = NDreduceSum(X, numberSpecies);
	
	/* Main time integration loop */
	if(plotOutput == 1) {
		printf("SO\n");//StartOutput
  }
	
	while (t < integrationData.t_max)
	{
		if(plotOutput == 1 && log10(t) >= plotStartTime) {
			//Do this once after log10(t) >= plotStartTime.
			if(setNextOut == 0) {
	      intervalLogt = (log10(integrationData.t_max)-log10(t))/numIntervals;
				nextOutput = log10(t);
				setNextOut = 1;
			}
		//stdout to file > fernOut.txt for plottable output
      //tolerance to check if time is close to nextOutput
      fern_real nextOuttol = fabs(log10(t)-nextOutput)/fabs(nextOutput);
			if(nextOuttol <= .000001) {
        //printf("tol = %f, log10(t): %f, nextOut: %f\n", nextOuttol, log10(t), nextOutput);
				printf("OC\n");//OutputCount
				//printf("OC: %d\n", outputCount);//OutputCount
				//renormalize nextOutput by compensating for overshooting last expected output time
				nextOutput = intervalLogt+nextOutput;
        //For this timestep start asy and pe counts at zero, then count them up for this timestep
				asyCount = 0;
				peCount = 0;
        //Check all Species if undergoing asymptotic update
				for(int m = 0; m < network.species; m++) {
				  printf("Y:%eZ:%dN:%dF+%eF-%e\n", Y[m], Z[m], N[m], Fplus[m], Fminus[m]);
					if(checkAsy(FminusSum[m], Y[m], dt)) {
						asyCount++;	
          }
				}
        //check frac RG PartialEq
	      partialEquil(Y, numberReactions, ReacGroups, network.reactant, network.product, final_k, pEquil, RGid, numRG, 0.01, eq);
        //Check all RG if in Equilibrium
				for(int i = 0; i < numRG; i++) {
					if(pEquil[i] == 1) {
						peCount++;			
          }
				}
				FracAsy = asyCount/numberSpecies;
				FracRGPE = peCount/numRG;
				printf("SUD\nti:%edt:%eT9:%erh:%esX:%efasy:%ffrpe:%f\n", t, dt, integrationData.T9, integrationData.rho, sumX, FracAsy, FracRGPE);//StartUniversalData
				outputCount++;
			}
		}
		/* Set Yzero[] to the values of Y[] updated in previous timestep. */
		
		for (int i = 0; i < numberSpecies; i++)
		{
			Yzero[i] = Y[i];
		}
		
		
		/* Compute the fluxes from the previously-computed rates and the current abundances. */
		
		/* Parallel version of flux calculation */
		
		for (int i = 0; i < numberReactions; i++)
		{
			int nr = network.numReactingSpecies[i];
			Flux[i] = Rate[i] * Y[network.reactant[0][i]];
			
			switch (nr)
			{
			case 3:
				/* 3-body; flux = rate x Y x Y x Y */
				Flux[i] *= Y[network.reactant[2][i]];
				
			case 2:
				/* 2-body; flux = rate x Y x Y */
				Flux[i] *= Y[network.reactant[1][i]];
				break;
			}
		}
		
		
		/* Populate the F+ and F- arrays in parallel from the master Flux array. */
		
		populateF(Fplus, FplusFac, Flux, MapFplus, totalFplus);
		populateF(Fminus, FminusFac, Flux, MapFminus, totalFminus);


		/*
		   Sum the F+ and F- for each isotope. These are "sub-arrays"
		   of Fplus and Fminus at (F[+ or -] + minny) of size FplusMax[i].
		   The first loop applies to sub-arrays with size < 40. The outer
		   loop (in i) is parallel, but the inner loops (in j) are serial.

		   Alpha particles, protons, and neutrons all have size much
		   greater than 40, and they are summed in the next for loop, which
		   uses the NDreduceSum.
		*/
		
		int minny;
		
		for (int i = 0; i < numberSpecies; i++)
		{
            minny = (i > 0) ? FplusMax[i - 1] + 1 : 0;
			/* Serially sum secction of F+. */
			FplusSum[i] = 0.0;
			for (int j = minny; j <= FplusMax[i]; j++)
			{
				FplusSum[i] += Fplus[j];
			}

			/* Serially sum section of F-. */
           	minny = (i > 0) ? FminusMax[i - 1] + 1 : 0;
			FminusSum[i] = 0.0;
			for (int j = minny; j <= FminusMax[i]; j++)
			{
				FminusSum[i] += Fminus[j];
			}
		}
		
		/* Find the maximum value of |FplusSum-FminusSum| to use in setting timestep. */
		
		for (int i = 0; i < numberSpecies; i++)
		{
			#ifdef FERN_SINGLE
				Fdiff[i] = fabsf(FplusSum[i] - FminusSum[i]);
			#else
				Fdiff[i] = fabs(FplusSum[i] - FminusSum[i]);
			#endif
		}
		
		
		/* Call tree algorithm to find max of array Fdiff. */

		maxFlux = reduceMax(Fdiff, numberSpecies);

		
		/*
		   Now use the fluxes to update the populations in parallel for this timestep.
		   For now we shall assume the asymptotic method. We determine whether each isotope
		   satisfies the asymptotic condition. If it does we update with the asymptotic formula.
		   If not, we update numerically using the forward Euler formula.
		*/
		
		/* Determine an initial trial timestep based on fluxes and dt in previous step. */
		
		dtFlux = fluxFrac / maxFlux;
		dtFloor = floorFac * t;
		if (dtFlux > dtFloor) dtFlux = dtFloor;
			
		dt = dtFlux;
		if (deltaTimeRestart < dtFlux) dt = deltaTimeRestart;
		
		updatePopulations(FplusSum, FminusSum, Y, Yzero, numberSpecies, dt);
		
		/* Compute sum of mass fractions sumX for all species. */
		
		for (int i = 0; i < numberSpecies; i++)
		{
			/* Compute mass fraction X from abundance Y. */
			X[i] = massNum[i] * Y[i];
		}
		
		sumX = NDreduceSum(X, numberSpecies);
		
		
		/*
		   Now modify timestep if necessary to ensure that particle number is conserved to
		   specified tolerance (but not too high a tolerance). Using updated populations
		   based on the trial timestep computed above, test for conservation of particle
		   number and modify trial timestep accordingly.
		*/
		
		#ifdef FERN_SINGLE
			fern_real test1 = fabsf(sumXLast - 1.0);
			fern_real test2 = fabsf(sumX - 1.0);
			massChecker = fabsf(sumXLast - sumX);

			if (test2 > test1 && massChecker > massTol)
			{
				dt *= fmaxf(massTol / fmaxf(massChecker, (fern_real) 1.0e-16), downbumper);
			}
			else if (massChecker < massTolUp)
			{
				dt *= (massTol / (fmaxf(massChecker, upbumper)));
			}
		#else
			fern_real test1 = fabs(sumXLast - 1.0);
			fern_real test2 = fabs(sumX - 1.0);
			massChecker = fabs(sumXLast - sumX);

			if (test2 > test1 && massChecker > massTol)
			{
				dt *= fmax(massTol / fmax(massChecker, (fern_real) 1.0e-16), downbumper);
			}
			else if (massChecker < massTolUp)
			{
				dt *= (massTol / (fmax(massChecker, upbumper)));
			}
		#endif
		
		
		updatePopulations(FplusSum, FminusSum, Y, Yzero, numberSpecies, dt);
		
		
		/*
		   Store the actual timestep that would be taken. Same as dt unless
		   artificially shortened in the last integration step to match end time.
		*/

		deltaTimeRestart = dt;

    /*
      DS plotting addon, if next time is greater than plotStartTime, set dt to 
      diff between plotStartTime and current time. This will ensure that the 
      output data begins at exactly the plotSartTime
    */ 

    if(plotOutput == 1 && log10(t+dt) > plotStartTime && setNextOut == 0) {
      #ifdef FERN_SINGLE
        dt = pow(10, plotStartTime) - t;
      #else
        dt = powf(10, plotStartTime) - t;
      #endif
    }

    if(plotOutput == 1 && log10(t+dt) > nextOutput) {
//      printf("nextOut: %e, t: %e, dt: %e, t+dt: %e", pow(10, nextOutput), t, dt, t+dt);
      #ifdef FERN_SINGLE
        dt = pow(10, nextOutput) - t;
      #else
        dt = powf(10, nextOutput) - t;
      #endif
 //     printf(", updated time: %e\n", t+dt);
    }
		
		/*
		   Finally check to be sure that timestep will not overstep next plot output
		   time and adjust to match if necessary. This will adjust dt only if at the end
		   of the integration interval. In that case it will also recompute the Y[]
		   corresponding to the adjusted time interval.
		 */
		
		if (t + dt >= integrationData.t_max)
		{
				/*
				   TODO
				   Copy back to CPU for dt_init next operator split integration.
				   Params2[2] = dt;
				*/
				
			dt = integrationData.t_max - t;
			
			
			updatePopulations(FplusSum, FminusSum, Y, Yzero, numberSpecies, dt);
		}
		
		
		/* NOTE: eventually need to deal with special case Be8 <-> 2 He4. */
		
		/* Now that final dt is set, compute final sum of mass fractions sumX. */
		
		for (int i = 0; i < numberSpecies; i++)
		{
			/* Compute mass fraction X from abundance Y. */
			X[i] = massNum[i] * Y[i];
		}
		
		sumX = NDreduceSum(X, numberSpecies);
		
		
		/* Increment the integration time and set the new timestep. */
			
		t += dt;
		timesteps++;
		
		sumXLast = sumX;
	}
	if(plotOutput == 1)
		printf("EO\n");//EndOutput
}


/* Device functions */

/*
   Determines whether an isotope specified by speciesIndex satisfies the
   asymptotic condition. Returns 1 if it does and 0 if not.
*/

inline bool checkAsy(fern_real Fminus, fern_real Y, fern_real dt)
{
	/* This is not needed because 1.0 / 0.0 == inf in C and inf > 1.0 */
	
	/*
	   Prevent division by zero in next step
	   if (Y == 0.0)
	     return false;
	*/
	
	return (Fminus * dt / Y > 1.0);
}


/* Returns the updated Y using the asymptotic formula */

inline fern_real asymptoticUpdate(fern_real Fplus, fern_real Fminus, fern_real Y, fern_real dt)
{
	/* Sophia He formula */
	return (Y + Fplus * dt) / (1.0 + Fminus * dt / Y);
}


/* Returns the Y specified by speciesIndex updated using the forward Euler method */

inline fern_real eulerUpdate(fern_real FplusSum, fern_real FminusSum, fern_real Y, fern_real dt)
{
	return Y + (FplusSum - FminusSum) * dt;
}

/*
   Non-destructive sum reduction.
   Same as previous, but copies array to dsmem allocated
   to the global scratch_space before executing algorithm.
*/

fern_real NDreduceSum(fern_real *a, unsigned short length)
{
	fern_real sum;

	sum = 0.0;

	for (int i = 0; i < length; i++) {
		sum += a[i];
	}

    return sum;
}

/*
   Performs a parallel maximum reduction in O(log(length)) time

   The given array is overwritten by intermediate values during computation.
   The maximum array size is 2 * blockDim.x.
*/
fern_real reduceMax(fern_real *a, unsigned short length)
{
    fern_real max;
    max = a[0];
    for (int i = 0; i < length; i++) {
        if (a[i] > max) {
            max = a[i];    
        }
    }
	
	return max;
}

/*
   Populates Fplus or Fminus
   Since the calculations for Fplus and Fminus are similar, the implementation
   of this function uses the term 'sign' to replace 'plus' and 'minus'.
*/

void populateF(fern_real *Fsign, fern_real *FsignFac, fern_real *Flux,
	unsigned short *MapFsign, unsigned short totalFsign)
{
	for (int i = 0; i < totalFsign; i++)
	{
		Fsign[i] = FsignFac[i] * Flux[MapFsign[i]];
	}
}


/* Updates populations based on the trial timestep */

inline void updatePopulations(fern_real *FplusSum, fern_real *FminusSum,
	fern_real *Y, fern_real *Yzero, unsigned short numberSpecies, fern_real dt)
{
	/* Parallel Update populations based on this trial timestep. */
	for (int i = 0; i < numberSpecies; i++)
	{
		if (checkAsy(FminusSum[i], Y[i], dt))
		{
			Y[i] = asymptoticUpdate(FplusSum[i], FminusSum[i], Yzero[i], dt);
		}
		else
		{
			Y[i] = eulerUpdate(FplusSum[i], FminusSum[i], Yzero[i], dt);
		}
	}
}

/* Checks for partial equilibrium between reaction groups */
void partialEquil(fern_real *Y, unsigned short numberReactions, int *ReacGroups, int **reactant, int **product, fern_real **final_k, int *pEquil, int *RGid, int numRG, fern_real tolerance, int eq) {
	fern_real y_a=0;
	fern_real y_b=0;
	fern_real y_c=0;
	fern_real y_d=0;
	fern_real y_e=0;
	fern_real c1=0;
	fern_real c2=0;
	fern_real c3=0;
	fern_real c4=0;
	fern_real a=0;
	fern_real b=0;
	fern_real c=0;
	fern_real alpha=0;
	fern_real beta=0;
	fern_real gamma=0;
	fern_real q=0;
	fern_real y_eq_a=0;
	fern_real y_eq_b=0;
	fern_real y_eq_c=0;
	fern_real y_eq_d=0;
	fern_real y_eq_e=0;
	fern_real PE_val_a=0;
	fern_real PE_val_b=0;
	fern_real PE_val_c=0;
	fern_real PE_val_d=0;
	fern_real PE_val_e=0;
  int members=0;
	bool PEprintData = false;

		//final partial equilibrium loop for calculating equilibrium
		for(int i = 0; i < numRG; i++) {
				if(PEprintData) {
          printf("RGNUM: %d, Class: %d, RGid: %d\n", i, ReacGroups[RGid[i]], RGid[i]);
        }
        pEquil[i] = 0;
	      //reset RG reactant and product populations
        y_a = 0;
        y_b = 0;
        y_c = 0;
        y_d = 0;
        y_e = 0;
        //if i is not the last RG, calculate number of members by the difference between
        //this RG's parent rID, and that of the next RG. Else, take the difference between the
        //total number of reactions and the current RG parent rID.
        if(i!=numRG-1) {
          members = RGid[i+1]-RGid[i];
        } else {
          members = numberReactions-RGid[i];
        }
				//Get current population for each reactant and product of this RG
				//TODO: figure out how to differentiate between a neutron as reactant/product and a null entry, as n has Isotope species ID = 0.
				//TODO: Something to watch out for: if a reaction has, for example, three reactants and two products such as in RGclass 5,
				// will it be presented first (RGParent) as a+b+c --> d+e, or might the RGParent have the reverse set up, a+b --> c+d+e
				// if the latter occurs, we'll need to add some logic to account for that. Right now, assuming that all RGParents are set up
				// in the former scenario. This would then be another instance where we'll need to differentiate between neutrons and null 
				// in the reactant and product arrays.
				if(ReacGroups[RGid[i]] == 1) {
					y_a = Y[reactant[0][RGid[i]]];
					y_b = Y[product[0][RGid[i]]];
          //set specific constraints and coefficients for RGclass 1
          c1 = y_a+y_b;
          a = 0;
          b = -final_k[0][i];
          c = final_k[1][i];
          q = 0;

          //theoretical equilibrium population of given species
          y_eq_a = -c/b;
          y_eq_b = c1-y_eq_a;

          //is each reactant and product in equilibrium?
          PE_val_a = fabs(y_a-y_eq_a)/fabs(y_eq_a);
          PE_val_b = fabs(y_b-y_eq_b)/fabs(y_eq_b);
          if(PE_val_a < tolerance && PE_val_b < tolerance) {
            pEquil[i] = 1;
          } 
					if(PEprintData) {
						printf("RG %d: members=%d kf=%e kr=%e\n RGClass=%d c1=%e\n a=%e b=%e c=%e\n q=%e\n y0_eq: %e y0: %e y1_eq: %e y1: %e R0: %e R1: %e inEquilibrium: %d\n\n",i,members,final_k[0][i],final_k[1][i],ReacGroups[RGid[i]], c1, a, b, c, q, y_eq_a, y_a, y_eq_b, y_b, PE_val_a, PE_val_b, pEquil[i]);
          }
				} 
				else if(ReacGroups[RGid[i]] == 2) {
          y_a = Y[reactant[0][RGid[i]]];
          y_b = Y[reactant[1][RGid[i]]];
          y_c = Y[product[0][RGid[i]]];
          c1 = y_b-y_a;
          c2 = y_b+y_c;
          a = -final_k[0][i];
          b = -(c1*final_k[0][i]+final_k[1][i]);
          c = final_k[1][i]*(c2-c1);
          q = (4*a*c)-(b*b);
          y_eq_a = ((-.5/a)*(b+sqrt(-q)));
          y_eq_b = y_eq_a+c1;
          y_eq_c = c2-y_eq_b;
          PE_val_a = fabs(y_a-y_eq_a)/fabs(y_eq_a);
          PE_val_b = fabs(y_b-y_eq_b)/fabs(y_eq_b);
          PE_val_c = fabs(y_c-y_eq_c)/fabs(y_eq_c);
          if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance) {
            pEquil[i] = 1;
          }   
          if(PEprintData) {
    				printf("RG %d: members=%d kf=%e kr=%e\n RGClass= %d c1=%e c2=%e\n a=%e b=%e c=%e\n q=%e\n y0_eq: %e y0: %e y1_eq: %e y1: %e y2_eq: %e y2: %e\nR0:%e R1: %e R2: %e inEquilibrium: %d\n\n",i, members,final_k[0][i],final_k[1][i],ReacGroups[RGid[i]], c1, c2, a, b, c, q, y_eq_a, y_a, y_eq_b, y_b, y_eq_c, y_c, PE_val_a, PE_val_b, PE_val_c, pEquil[i]);
          }
        }
        else if(ReacGroups[RGid[i]] == 3) {
          y_a = Y[reactant[0][RGid[i]]];
          y_b = Y[reactant[1][RGid[i]]];
          y_c = Y[reactant[2][RGid[i]]];
          y_d = Y[product[0][RGid[i]]];
          c1 = y_a-y_b;
          c2 = y_a-y_c;
          c3 = ((y_a+y_b+y_c)/3)+y_d;
          a = final_k[0][i]*(c1+c2)-final_k[0][i]*y_a;
          b = -((final_k[0][i]*c1*c2)+final_k[1][i]);
          c = final_k[1][i]*(c3+(c1/3)+(c2/3));
          q = (4*a*c)-(b*b);
          y_eq_a = ((-.5/a)*(b+sqrt(-q)));
          y_eq_b = y_eq_a-c1;
          y_eq_c = y_eq_a-c2;
          y_eq_d = c3-y_eq_a+((1/3)*(c1+c2));
          PE_val_a = fabs(y_a-y_eq_a)/fabs(y_eq_a);
          PE_val_b = fabs(y_b-y_eq_b)/fabs(y_eq_b);
          PE_val_c = fabs(y_c-y_eq_c)/fabs(y_eq_c);
          PE_val_d = fabs(y_d-y_eq_d)/fabs(y_eq_d);
          if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance && PE_val_d < tolerance) {
            pEquil[i] = 1;
          } 
          if(PEprintData) {
 						printf("RG %d: members=%d kf=%e kr=%e\n RGClass= %d c1=%e c2=%e c3=%e\n a=%e b=%e c=%e\n q=%e\n y0_eq: %e y0: %e y1_eq: %e y1: %e y2_eq: %e y2:%e y3_eq: %e y3: %e\nR0: %e R1: %e R2: %e R3: %e, inEquilibrium: %d\n\n",i, members,final_k[0][i],final_k[1][i],ReacGroups[RGid[i]], c1, c2, c3, a, b, c, q, y_eq_a, y_a, y_eq_b, y_b, y_eq_c, y_c, y_eq_d, y_d, PE_val_a, PE_val_b, PE_val_c, PE_val_d, pEquil[i]);
          }
        }
        else if(ReacGroups[RGid[i]] == 4) {
          y_a = Y[reactant[0][RGid[i]]];
          y_b = Y[reactant[1][RGid[i]]];
          y_c = Y[product[0][RGid[i]]];
          y_d = Y[product[1][RGid[i]]];
          c1 = y_a-y_b;
          c2 = y_a+y_c;
          c3 = y_a+y_d;
          a = final_k[1][i]-final_k[0][i];
          b = -(final_k[1][i]*(c2+c3))+(final_k[0][i]*c1);
          c = final_k[1][i]*c2*c3;
 					q = (4*a*c)-(b*b);
          y_eq_a = ((-.5/a)*(b+sqrt(-q)));	
  				y_eq_b = y_eq_a-c1;
  				y_eq_c = c2-y_eq_a;
  				y_eq_d = c3-y_eq_a;
	        PE_val_a = fabs(y_a-y_eq_a)/fabs(y_eq_a);
  				PE_val_b = fabs(y_b-y_eq_b)/fabs(y_eq_b);
	        PE_val_c = fabs(y_c-y_eq_c)/fabs(y_eq_c);
  				PE_val_d = fabs(y_d-y_eq_d)/fabs(y_eq_d);
          if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance && PE_val_d < tolerance) {
    				pEquil[i] = 1;
		      }
          if(PEprintData) {
    				printf("RG %d: members=%d kf=%e kr=%e\n RGClass= %d c1=%e c2=%e c3=%e\n a=%e b=%e c=%e\n q=%e\n y0_eq: %e y0: %e y1_eq: %e y1: %e y2_eq: %e y2: %e y3_eq: %e y3: %e\nR0: %e R1: %e R2: %e R3: %e inEquilibrium: %d\n\n",i, members,final_k[0][i],final_k[1][i],ReacGroups[RGid[i]], c1, c2, c3, a, b, c, q, y_eq_a, y_a, y_eq_b, y_b, y_eq_c, y_c, y_eq_d, y_d, PE_val_a, PE_val_b, PE_val_c, PE_val_d, pEquil[i]);
          }
		    }
      	else if(ReacGroups[RGid[i]] == 5) {
          y_a = Y[reactant[0][RGid[i]]];
          y_b = Y[reactant[1][RGid[i]]];
          y_c = Y[product[0][RGid[i]]];
          y_d = Y[product[1][RGid[i]]];
			   	y_e = Y[product[2][RGid[i]]];
          c1 = y_a+((y_c+y_d+y_e)/3);
          c2 = y_a-y_b;
          c3 = y_c-y_d;
          c4 = y_c-y_e;
          a = (((3*c1)-y_a)*final_k[1][i])-final_k[0][i];
 					alpha = c1+((c3+c4)/3);	
          beta = c1-(2*c3/3)+(c4/3);	
  				gamma = c1+(c3/3)-(2*c4/3);	
          b = (c2*final_k[0][i])-(((alpha*beta)+(alpha*gamma)+(beta*gamma))*final_k[1][i]);
          c = final_k[1][i]*alpha*beta*gamma;
          q = (4*a*c)-(b*b);
          y_eq_a = ((-.5/a)*(b+sqrt(-q)));
          y_eq_b = y_eq_a-c2;
          y_eq_c = alpha-y_eq_a;
          y_eq_d = beta-y_eq_a;
          y_eq_e = gamma-y_eq_a;
          PE_val_a = fabs(y_a-y_eq_a)/fabs(y_eq_a);
          PE_val_b = fabs(y_b-y_eq_b)/fabs(y_eq_b);
          PE_val_c = fabs(y_c-y_eq_c)/fabs(y_eq_c);
          PE_val_d = fabs(y_d-y_eq_d)/fabs(y_eq_d);
          PE_val_e = fabs(y_e-y_eq_e)/fabs(y_eq_e);
          if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance && PE_val_d < tolerance && PE_val_e < tolerance) {
            pEquil[i] = 1;
          } 
          if(PEprintData) {
  					printf("RG %d: members=%d kf=%e kr=%e\n RGClass= %d c1=%e c2=%e c3=%e c4=%e\n a=%e b=%e c=%e\n q=%e\n y0_eq: %e y0: %e y1_eq: %e y1: %e y2_eq: %e y2: %e y3_eq: %e y3: %e y4_eq: %e y4: %e\nR0: %e R1: %e R2: %e R3: %e R4:%e inEquilibrium: %d\n\n",i, members,final_k[0][i],final_k[1][i],ReacGroups[RGid[i]], c1, c2, c3, c4, a, b, c, q, y_eq_a, y_a, y_eq_b, y_b, y_eq_c, y_c, y_eq_d, y_d, y_eq_e, y_e, PE_val_a, PE_val_b, PE_val_c, PE_val_d, PE_val_e, pEquil[i]); 
          }
	      }
		}
}

void network_print(const Network &network)
{
	/* Network data */
	
	printf("species: %d\n", network.species);
	
	printf("Z: { ");
	for (int i = 0; i < network.species; i++)
		printf("%4d ", network.Z[i]);
	printf("}\n");
	
	printf("N: { ");
	for (int i = 0; i < network.species; i++)
		printf("%4d ", network.N[i]);
	printf("}\n");
	
	/* Reaction data */
	
	printf("\n");
	
	printf("reactions: %d\n", network.reactions);
	
	for (int n = 0; n < 7; n++)
	{
		printf("P[%d]: { ", n);
		for (int i = 0; i < network.reactions; i++)
			printf("%e ", network.P[n][i]);;
		printf("\n");
	}
	
	printf("numReactingSpecies: { ");
	for (int i = 0; i < network.reactions; i++)
		printf("%4d ", network.numReactingSpecies[i]);
	printf("}\n");
	
	
	printf("statFac: { ");
	for (int i = 0; i < network.reactions; i++)
		printf("%e ", network.statFac[i]);
	printf("}\n");
	
	
	printf("Q: { ");
	for (int i = 0; i < network.reactions; i++)
		printf("%e ", network.Q[i]);
	printf("}\n");
	
	for (int n = 0; n < 3; n++)
	{
		printf("reactant[%d]: { ", n);
		for (int i = 0; i < network.reactions; i++)
			printf("%4d ", network.reactant[n][i]);
		printf("}\n");
	}
	
	printf("totalFplus: %d\n", network.totalFplus);
	printf("totalFminus: %d\n", network.totalFminus);
	
	printf("FplusFac: { ");
	for (int i = 0; i < network.totalFplus; i++)
		printf("%e ", network.FplusFac[i]);
	printf("}\n");
	
	printf("FminusFac: { ");
	for (int i = 0; i < network.totalFminus; i++)
		printf("%e ", network.FminusFac[i]);
	printf("}\n");
	
	printf("MapFplus: { ");
	for (int i = 0; i < network.totalFplus; i++)
		printf("%4u ", network.MapFplus[i]);
	printf("}\n");
	
	printf("MapFminus: { ");
	for (int i = 0; i < network.totalFminus; i++)
		printf("%4u ", network.MapFminus[i]);
	printf("}\n");
	
	printf("FplusMax: { ");
	for (int i = 0; i < network.species; i++)
		printf("%4u ", network.FplusMax[i]);
	printf("}\n");
	
	printf("FminusMax: { ");
	for (int i = 0; i < network.species; i++)
		printf("%4u ", network.FminusMax[i]);
	printf("}\n");
}
