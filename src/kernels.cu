#include <stdio.h>
#include "kernels.cuh"

extern __shared__ char dsmem[];
__device__ fern_real *scratch_space;

__global__ void integrateNetwork(
	Network network,
	IntegrationData integrationData,
	Globals *globalsPtr
)
{
	const int tid = threadIdx.x;
	
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

	/* Declare pointer variables for Partial Equilibrium arrays */
    int *ReacParent = new int [numberReactions];
	/* Declare pointer variables for IntegrationData arrays.  */

	fern_real *Y;

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

	/* Assign Partial Equilibrium pointers. */

	/* Assign IntegrationData pointers. */
	
	Y = integrationData.Y;

	/*
	   TODO
	   Evaluate shared bank conflicts to avoid serializing the shared
	   memory accesses.

	   Ref: https://developer.nvidia.com/content/using-shared-memory-cuda-cc
	*/

	/* Allocate shared memory. */

	int shared_pos;

	shared_pos = 0;

	FplusSum = (fern_real *) (dsmem + shared_pos);
	shared_pos += network.species * sizeof(fern_real);
	FminusSum = (fern_real *) (dsmem + shared_pos);
	shared_pos += network.species * sizeof(fern_real);

	/*
	   Allocate dsmem scratch space (see NDreduceSum).
	   To be safe, ensure numberReactions * sizeof(fern_real)
	   bytes are available, although this can be trimmed
	   quite a bit for production.
	*/

	scratch_space = (fern_real *) (dsmem + shared_pos);
	shared_pos += network.reactions * sizeof(fern_real);

	if (tid == 0) printf("%d bytes of dsmem used.\n", shared_pos);

	__syncthreads();


	/* Static shared memory */
	
	__shared__ fern_real maxFlux;
	__shared__ fern_real sumX;
	__shared__ fern_real t;
	__shared__ fern_real dt;
	__shared__ unsigned int timesteps;
	
	fern_real sumXLast;

	/* Compute the preFac vector. */
	
	if (tid == 0)
	{
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
	}

	/* Compute the rate values. */

	if (tid == 0)
	{
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
	}

	/* Author: Daniel Shyles */
	/* Begin Partial Equilibrium calculation */
    int RGParent;
	const bool displayRGdata = true;
    fern_real y_a;
    fern_real y_b;
    fern_real y_c;
    fern_real y_d;
    fern_real y_e;
	//constants for coefficient and relative abundance calculations
	fern_real c1;
	fern_real c2;
	fern_real c3;
	fern_real c4;
	//TODO: Add this back in -->	fern_real c4 = y_a+y_e; 
	//coefficients for q and theoretical equilibrium abundance calculation
	fern_real a;
	fern_real b;
	fern_real c;
	//to be used in RG Class 5 PE calculation;
	fern_real alpha;
	fern_real beta;
	fern_real gamma;
	fern_real q;
	fern_real y_eq_a; 
	fern_real y_eq_b; 
	fern_real y_eq_c; 
	fern_real y_eq_d; 
	fern_real y_eq_e; 
	//set tolerance for determining partial equilibrium
	fern_real tolerance = .01;
	//result of PE determination... if PE_val < tolerance, we're in equilibrium
	fern_real PE_val_a;
	fern_real PE_val_b;
	fern_real PE_val_c;
	fern_real PE_val_d;
	fern_real PE_val_e;
	//if in equilibrium I'll set this to 1
	int eq;

	// Population at time t (unnecessary for PE calculation as we have current populations):
	//fern_real y_0;
    //fern_real phi = ((2*a*y_0)+b+sqrt(-q))/((2*a*y_0)+b-sqrt(-q));
	//fern_real y_it = (-.5/a)*(b+(sqrt(-q)*((1+phi*expf(-sqrt(-q)*t))/(1-phi*expf(-sqrt(-q)*t)))));

	if (tid == 0) {
		//first loop through reactions to set up RG parents
		int numRG = 0;
		for(int i = 0; i < network.reactions; i++) {
		    if(network.ReacGroups[i] != 0) {
				numRG++;
			    //This indicates a new reaction group
				RGParent = i;
		    }
	        ReacParent[i] = RGParent;
		}
	    fern_real kf;
		fern_real kr;
	    fern_real *final_k[2];
		//set up array of final reaction rates for each RG
		//doing it this way saves memory, as we don't need two arrays with size numberReactions, only numRG
		for(int m = 0; m < 2; m++)
			final_k[m] = new fern_real [numRG];
		if(displayRGdata)
			printf("Start Reaction Group Data\nNumber Reaction Groups: %d\n\n",numRG);
		//second to calculate final reaciton rates for each RG
		for(int i = 0; i < network.reactions; i++) {

			if(displayRGdata && network.ReacGroups[i] != 0) {
				printf("RG Class: %d\nRG ID (Parent): %d\n", network.ReacGroups[i], i);
				//output numReactingSpecies and numProducts for Parent of RG
                printf("numReacting: %d, numProducts: %d\n", network.numReactingSpecies[i], network.PEnumProducts[i]);			
				if(network.ReacGroups[i] == 1) {		
					printf("Reactant SID: %d; Product SID: %d\n",network.reactant[0][i], network.product[0][i]);
				} 
				else if(network.ReacGroups[i] == 2) {
                    printf("Reactant SID: %d, %d; Product SID: %d\n",network.reactant[0][i], network.reactant[1][i], network.product[0][i]);
                }
                else if(network.ReacGroups[i] == 3) {
                    printf("Reactant SID: %d, %d, %d; Product SID: %d\n",network.reactant[0][i], network.reactant[1][i], network.reactant[2][i], network.product[0][i]);
                }
                else if(network.ReacGroups[i] == 4) {
                    printf("Reactant SID: %d, %d; Product SID: %d, %d\n",network.reactant[0][i], network.reactant[1][i], network.product[0][i], network.product[1][i]);
                }
                else if(network.ReacGroups[i] == 5) {
                    printf("Reactant SID: %d, %d; Product SID: %d, %d, %d\n",network.reactant[0][i], network.reactant[1][i], network.product[0][i], network.product[1][i], network.product[2][i]);
                }
				printf("-----\n|\n");
			}				
			if(displayRGdata)
				printf("Reaction ID: %d\nRG Member ID: %d\nForward Reaction (+-Q): %d\nRate: %f\n|\n", i, network.RGmemberIndex[i], network.pnQ[i], Rate[i]);

            //if RGmemberindex is greater (or equal for RGmemberindex[i] = RGmemberindex[i+1] = 0 than next one, then end of Reaction Group
            if(network.RGmemberIndex[i] >= network.RGmemberIndex[i+1]) {
                //get forward and reverse rates for all reactions within group, starting with i-network.RGmemberIndex[i], and ending with i.
                kf = 0; //forward rate
                kr = 0; //reverse rate
                //iterate through each RGmember and calculate the total rate from forward and reverse reactions
                for(int n = network.RGmemberIndex[i]; n >= 0; n--) {
                    //add the rate to forward reaction
                    /****************************************************************************************************************************
                     *TODO rework this such that reactions with multiple sets of parameters (n > 1, three or more reactions per reaction group) *
                     *have their final rates (kf, kr) calculated properly. Currently being summed, but some other operation is necessary here...*
                     ****************************************************************************************************************************/
                    if(network.pnQ[i-n] == 1) {
                        kf += Rate[i-n];
                    } else {
                    //add the rate to reverse reaction
                        kr += Rate[i-n];
                    }
                    //printf("current forward Rate for RG with Parent %d = %f and kf = %f \n ", ReacParent[i], Rate[i-n], kf);
                    //printf("current reverse Rate for RG with Parent %d = %f and kr = %f \n ", ReacParent[i], Rate[i-n], kr);
                }
                final_k[0][ReacParent[i]] = kf;
                final_k[1][ReacParent[i]] = kr;
                if(displayRGdata) {
                    printf("-----\n");
                    printf("Final Forward Rate: kf = %f \n", final_k[0][ReacParent[i]]);
                    printf("Final Reverse Rate: kr = %f \n", final_k[1][ReacParent[i]]);
                    printf("\n\n\n");
                }
            }
		}

		//final partial equilibrium loop for calculating equilibrium
		const bool displayPEdata = true;
		if(displayPEdata)
			printf("Start Partial Equilibrium Data\n");		
		for(int i = 0; i < numberReactions; i++) {
	        //reset RG reactant and product populations
            y_a = 0;
            y_b = 0;
            y_c = 0;
            y_d = 0;
            y_e = 0;
			if(network.ReacGroups[i] !=0) {
				//Get current population for each reactant and product of this RG
				//TODO: figure out how to differentiate between a neutron as reactant/product and a null entry, as n has Isotope species ID = 0.
				//TODO: Something to watch out for: if a reaction has, for example, three reactants and two products such as in RGclass 5,
				// will it be presented first (RGParent) as a+b+c --> d+e, or might the RGParent have the reverse set up, a+b --> c+d+e
				// if the latter occurs, we'll need to add some logic to account for that. Right now, assuming that all RGParents are set up
				// in the former scenario. This would then be another instance where we'll need to differentiate between neutrons and null 
				// in the reactant and product arrays.
				if(network.ReacGroups[i] == 1) {
					y_a = Y[network.reactant[0][i]];
					y_b = Y[network.product[0][i]];
                   //set specific constraints and coefficients for RGclass 2
                    c1 = y_a+y_b;
                    a = 0;
                    b = -final_k[0][i];
                    c = final_k[1][i];
                    q = 0;

                    //theoretical equilibrium population of given species
                    y_eq_a = -c/b;
                    y_eq_b = c1-y_eq_a;

                    //is each reactant and product in equilibrium?
                    PE_val_a = abs(y_a-y_eq_a)/(y_eq_a);
                    PE_val_b = abs(y_b-y_eq_b)/(y_eq_b);
                    if(PE_val_a < tolerance && PE_val_b < tolerance) {
                        eq = 1;
                    } else {
                        eq = 0;
                    }

                    if(displayPEdata && !isnan(PE_val_a) && !isnan(PE_val_b)) {
                        //only print if there are populations with which to calculate
                        printf("Reaction Group ID: %d, RG Class: %d\n",i, network.ReacGroups[i]);
                        for (int n = 0; n < 2; n++)
                            printf("Reactant[%d]: %d\n",n, network.reactant[n][i]);
                        for (int n = 0; n < 2; n++)
                            printf("Product[%d]: %d\n",n, network.product[n][i]);

                        printf("PE_val for y_a (Species: %d): %f\nPE_val for y_b (Species: %d): %f\nIs this RG in equilibrium?: %d\n",network.reactant[0][i],PE_val_a,network.product[0][i],PE_val_b,eq);
                        printf("\n\n");
                    }
				} 
				else if(network.ReacGroups[i] == 2) {
                    y_a = Y[network.reactant[0][i]];
                    y_b = Y[network.reactant[1][i]];
                    y_c = Y[network.product[0][i]];
                    //set specific constraints and coefficients for RGclass 2
                    c1 = y_b-y_a;
                    c2 = y_b+y_c;
                    a = -final_k[0][i];
                    b = -(c1*final_k[0][i]+final_k[1][i]);
                    c = final_k[1][i]*(c2-c1);
                    q = (4*a*c)-(b*b);

                    //theoretical equilibrium population of given species
                    y_eq_a = ((-.5/a)*(b+sqrt(-q)));
                    y_eq_b = y_eq_a+c1;
                    y_eq_c = c2-y_eq_b;

                    //is each reactant and product in equilibrium?
                    PE_val_a = abs(y_a-y_eq_a)/(y_eq_a);
                    PE_val_b = abs(y_b-y_eq_b)/(y_eq_b);
                    PE_val_c = abs(y_c-y_eq_c)/(y_eq_c);
                    if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance) {
                        eq = 1;
                    } else {
                        eq = 0;
                    }

                    if(displayPEdata && !isnan(PE_val_a) && !isnan(PE_val_b) && !isnan(PE_val_c)) {
                        //only print if there are populations with which to calculate
                        printf("Reaction Group ID: %d, RG Class: %d\n",i, network.ReacGroups[i]);
                        for (int n = 0; n < 2; n++)
                            printf("Reactant[%d]: %d\n",n, network.reactant[n][i]);
                        for (int n = 0; n < 2; n++)
                            printf("Product[%d]: %d\n",n, network.product[n][i]);

						printf("PE_val for y_a (Species: %d): %f\nPE_val for y_b (Species: %d): %f\nPE_val for y_c (Species: %d): %f\nIs this RG in equilibrium?: %d\n",
							network.reactant[0][i],PE_val_a,
							network.reactant[1][i],PE_val_b,
							network.product[1][i],PE_val_c,
							eq
						);
                        printf("\n\n");
                    }
                }
                else if(network.ReacGroups[i] == 3) {
                    y_a = Y[network.reactant[0][i]];
                    y_b = Y[network.reactant[1][i]];
                    y_c = Y[network.reactant[2][i]];
                    y_d = Y[network.product[0][i]];
                    //set specific constraints and coefficients for RGclass 3
                    c1 = y_a-y_b;
                    c2 = y_a-y_c;
                    c3 = ((1/3)*(y_a+y_b+y_c))+y_d;
                    a = final_k[0][i]*(c1+c2)-final_k[0][i]*y_a;
                    b = -((final_k[0][i]*c1*c2)+final_k[1][i]);
                    c = final_k[1][i]*(c3+(c1/3)+(c2/3));
                    q = (4*a*c)-(b*b);

                    //theoretical equilibrium population
                    y_eq_a = ((-.5/a)*(b+sqrt(-q)));
                    y_eq_b = y_eq_a-c1;
                    y_eq_c = y_eq_a-c2;
                    y_eq_d = c3-y_eq_a+((1/3)*(c1+c2));

                    //is each reactant and product in equilibrium?
                    PE_val_a = abs(y_a-y_eq_a)/(y_eq_a);
                    PE_val_b = abs(y_b-y_eq_b)/(y_eq_b);
                    PE_val_c = abs(y_c-y_eq_c)/(y_eq_c);
                    PE_val_d = abs(y_d-y_eq_d)/(y_eq_d);
                    if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance && PE_val_d < tolerance) {
                        eq = 1;
                    } else {
                        eq = 0;
                    }

                    if(displayPEdata && !isnan(PE_val_a) && !isnan(PE_val_b) && !isnan(PE_val_c) && !isnan(PE_val_d)) {
                        //only print if there are populations with which to calculate
                        printf("Reaction Group ID: %d, RG Class: %d\n",i, network.ReacGroups[i]);
                        for (int n = 0; n < 2; n++)
                            printf("Reactant[%d]: %d\n",n, network.reactant[n][i]);
                        for (int n = 0; n < 2; n++)
                            printf("Product[%d]: %d\n",n, network.product[n][i]);

                        printf("PE_val for y_a (Species: %d): %f\nPE_val for y_b (Species: %d): %f\nPE_val for y_c (Species: %d): %f\nPE_val for y_d (Species: %d): %f\nIs this RG in equilibrium?: %d\n",
                            network.reactant[0][i],PE_val_a,
                            network.reactant[1][i],PE_val_b,
                            network.reactant[2][i],PE_val_c,
                            network.product[0][i],PE_val_d,
                            eq
                        );
                        printf("\n\n");
                    }

                }
                else if(network.ReacGroups[i] == 4) {
                    y_a = Y[network.reactant[0][i]];
                    y_b = Y[network.reactant[1][i]];
                    y_c = Y[network.product[0][i]];
                    y_d = Y[network.product[1][i]];

                    //set specific constraints and coefficients for RGclass 4
                    c1 = y_a-y_b;
                    c2 = y_a+y_c;
                    c3 = y_a+y_d;
                    a = final_k[1][i]-final_k[0][i];
                    b = -(final_k[1][i]*(c2+c3))+(final_k[0][i]*c1);
                    c = final_k[1][i]*c2*c3;
					q = (4*a*c)-(b*b);

					//calculate theoretical equilibrium value
					y_eq_a = ((-.5/a)*(b+sqrt(-q)));	
					y_eq_b = y_eq_a-c1;
					y_eq_c = c2-y_eq_a;
					y_eq_d = c3-y_eq_a;
				
					//is each reactant and product in equilibrium?	
					PE_val_a = abs(y_a-y_eq_a)/(y_eq_a);
					PE_val_b = abs(y_b-y_eq_b)/(y_eq_b);
					PE_val_c = abs(y_c-y_eq_c)/(y_eq_c);
					PE_val_d = abs(y_d-y_eq_d)/(y_eq_d);
					if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance && PE_val_d < tolerance) {
						eq = 1;
					} else {
						eq = 0;
					}

					if(displayPEdata && !isnan(PE_val_a) && !isnan(PE_val_b) && !isnan(PE_val_c) && !isnan(PE_val_d)) {
                        //only print if there are populations with which to calculate
						printf("Reaction Group ID: %d, RG Class: %d\n",i, network.ReacGroups[i]);
						for (int n = 0; n < 2; n++)
							printf("Reactant[%d]: %d\n",n, network.reactant[n][i]);
						for (int n = 0; n < 2; n++)
							printf("Product[%d]: %d\n",n, network.product[n][i]);

                        printf("PE_val for y_a (Species: %d): %f\nPE_val for y_b (Species: %d): %f\nPE_val for y_c (Species: %d): %f\nPE_val for y_d (Species: %d): %f\nIs this RG in equilibrium?: %d\n",
                            network.reactant[0][i],PE_val_a,
                            network.reactant[1][i],PE_val_b,
                            network.product[0][i],PE_val_c,
                            network.product[1][i],PE_val_d,
                            eq
                        );
						printf("\n\n");
					}
				}
				else if(network.ReacGroups[i] == 5) {
                    y_a = Y[network.reactant[0][i]];
                    y_b = Y[network.reactant[1][i]];
                    y_c = Y[network.product[0][i]];
                    y_d = Y[network.product[1][i]];
					y_e = Y[network.product[2][i]];

                    //set specific constraints and coefficients for RGclass 5
                    c1 = y_a+(1/3)*(y_c+y_d+y_e);
                    c2 = y_a-y_b;
                    c3 = y_c-y_d;
                    c4 = y_c-y_e;
                    a = (((3*c1)-y_a)*final_k[1][i])-final_k[0][i];
					alpha = c1+((1/3)*(c3+c4));	
					beta = c1-(2*c3/3)+(c4/3);	
					gamma = c1+(c3/3)-(2*c4/3);	
                    b = -(c2*final_k[0][i])-(((alpha*beta)+(alpha*gamma)+(beta*gamma))*final_k[1][i]);
                    c = final_k[1][i]*alpha*beta*gamma;
                    q = (4*a*c)-(b*b);

					//calculate theoretical equilibrium values
                    y_eq_a = ((-.5/a)*(b+sqrt(-q)));
                    y_eq_b = y_eq_a-c2;
                    y_eq_c = alpha-y_eq_a;
                    y_eq_d = beta-y_eq_a;
                    y_eq_e = gamma-y_eq_a;

                    //is each reactant and product in equilibrium?
                    PE_val_a = abs(y_a-y_eq_a)/(y_eq_a);
                    PE_val_b = abs(y_b-y_eq_b)/(y_eq_b);
                    PE_val_c = abs(y_c-y_eq_c)/(y_eq_c);
                    PE_val_d = abs(y_d-y_eq_d)/(y_eq_d);
                    PE_val_e = abs(y_e-y_eq_e)/(y_eq_e);
                    if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance && PE_val_d < tolerance && PE_val_e < tolerance) {
                        eq = 1;
                    } else {
                        eq = 0;
                    }

                    if(displayPEdata && !isnan(PE_val_a) && !isnan(PE_val_b) && !isnan(PE_val_c) && !isnan(PE_val_d) && !isnan(PE_val_e)) {
                        //only print if there are populations with which to calculate
                        printf("Reaction Group ID: %d, RG Class: %d\n",i, network.ReacGroups[i]);
                        for (int n = 0; n < 2; n++)
                            printf("Reactant[%d]: %d\n",n, network.reactant[n][i]);
                        for (int n = 0; n < 2; n++)
                            printf("Product[%d]: %d\n",n, network.product[n][i]);

                        printf("PE_val for y_a (Species: %d): %f\nPE_val for y_b (Species: %d): %f\nPE_val for y_c (Species: %d): %f\nPE_val for y_d (Species: %d): %f\nPE_val for y_e (Species: %d): %f\nIs this RG in equilibrium?: %d\n",
                            network.reactant[0][i],PE_val_a,
                            network.reactant[1][i],PE_val_b,
                            network.product[0][i],PE_val_c,
                            network.product[1][i],PE_val_d,
                            network.product[2][i],PE_val_e,
                            eq
                        );
                        printf("\n\n");
                    }
			
				}
				//reactions between 146 and 155 have carbon 12 and oxygen 16, and have non-zero starting populations
				//if (i > 146 && i < 155) 
				//if(displayPEdata)
					//printf("\nReactant/Product populations:\ny_a: %e\ny_b: %e\ny_c: %e\ny_d: %e\ny_e: %e\n\n", y_a, y_b, y_c, y_d, y_e);
			}
		}
	}

	/***End Partial Equilibrium***/

	/*
	   Begin the time integration from t=0 to tmax. Rather than t=0 we
	   start at some very small value of t. This is required for the CUDA C
	   code as well as the Java version.
	*/
	
	if (tid == 0)
	{
		t = 1.0e-20;
		dt = integrationData.dt_init;
		timesteps = 1;
	}
	
	fern_real floorFac = 0.1;
	fern_real upbumper = 0.9 * massTol;
	fern_real downbumper = 0.1;
	fern_real massTolUp = 0.25 * massTol;
	fern_real deltaTimeRestart = dt;
	fern_real dtFloor;
	fern_real dtFlux;
	fern_real massChecker;
	
	/* Compute mass numbers and initial mass fractions X for all isotopes. */
	
	for (int i = tid; i < numberSpecies; i += blockDim.x)
	{
		massNum[i] = (fern_real) Z[i] + (fern_real) N[i];
		/* Compute mass fraction X from abundance Y. */
		X[i] = massNum[i] * Y[i];
	}
	
	__syncthreads();
	sumXLast = NDreduceSum(X, numberSpecies);
	
	/* Main time integration loop */
	
	while (t < integrationData.t_max)
	{
		__syncthreads();
		/* Set Yzero[] to the values of Y[] updated in previous timestep. */
		
		for (int i = tid; i < numberSpecies; i += blockDim.x)
		{
			Yzero[i] = Y[i];
		}
		
		__syncthreads();
		
		/* Compute the fluxes from the previously-computed rates and the current abundances. */
		
		/* Parallel version of flux calculation */
		
		for (int i = tid; i < numberReactions; i += blockDim.x)
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
		
		__syncthreads();
		
		/* Populate the F+ and F- arrays in parallel from the master Flux array. */
		
		populateF(Fplus, FplusFac, Flux, MapFplus, totalFplus);
		populateF(Fminus, FminusFac, Flux, MapFminus, totalFminus);

		__syncthreads();

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
		
		for (int i = tid; i < numberSpecies; i += blockDim.x)
		{
            minny = (i > 0) ? FplusMax[i - 1] + 1 : 0;
			if ((FplusMax[i] + 1) - minny < 40)
			{
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
		}
		
		for (int i = 0; i < numberSpecies; i++)
		{
            minny = (i > 0) ? FplusMax[i - 1] + 1 : 0;
			if ((FplusMax[i] + 1) - minny >= 40)
			{
				FplusSum[i] = NDreduceSum(Fplus + minny, (FplusMax[i] + 1) - minny);

            	minny = (i > 0) ? FminusMax[i - 1] + 1 : 0;
				FminusSum[i] = NDreduceSum(Fminus + minny, (FminusMax[i] + 1) - minny);
			}
		}

		__syncthreads();
		
		/* Find the maximum value of |FplusSum-FminusSum| to use in setting timestep. */
		
		for (int i = tid; i < numberSpecies; i += blockDim.x)
		{
			#ifdef FERN_SINGLE
				Fdiff[i] = fabsf(FplusSum[i] - FminusSum[i]);
			#else
				Fdiff[i] = fabs(FplusSum[i] - FminusSum[i]);
			#endif
		}
		
		__syncthreads();
		
		/* Call tree algorithm to find max of array Fdiff. */

		maxFlux = reduceMax(Fdiff, numberSpecies);

		__syncthreads();
		
		/*
		   Now use the fluxes to update the populations in parallel for this timestep.
		   For now we shall assume the asymptotic method. We determine whether each isotope
		   satisfies the asymptotic condition. If it does we update with the asymptotic formula.
		   If not, we update numerically using the forward Euler formula.
		*/
		
		/* Determine an initial trial timestep based on fluxes and dt in previous step. */
		
		if (tid == 0)
		{
			dtFlux = fluxFrac / maxFlux;
			dtFloor = floorFac * t;
			if (dtFlux > dtFloor) dtFlux = dtFloor;
			
			dt = dtFlux;
			if (deltaTimeRestart < dtFlux) dt = deltaTimeRestart;
		}
		
		__syncthreads();
		updatePopulations(FplusSum, FminusSum, Y, Yzero, numberSpecies, dt);
		__syncthreads();
		
		/* Compute sum of mass fractions sumX for all species. */
		
		for (int i = tid; i < numberSpecies; i += blockDim.x)
		{
			/* Compute mass fraction X from abundance Y. */
			X[i] = massNum[i] * Y[i];
		}
		
		__syncthreads();
		sumX = NDreduceSum(X, numberSpecies);
		
		__syncthreads();
		
		/*
		   Now modify timestep if necessary to ensure that particle number is conserved to
		   specified tolerance (but not too high a tolerance). Using updated populations
		   based on the trial timestep computed above, test for conservation of particle
		   number and modify trial timestep accordingly.
		*/
		
		if (tid == 0)
		{
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
		}
		
		__syncthreads();
		
		updatePopulations(FplusSum, FminusSum, Y, Yzero, numberSpecies, dt);
		
		__syncthreads();
		
		/*
		   Store the actual timestep that would be taken. Same as dt unless
		   artificially shortened in the last integration step to match end time.
		*/
		
		deltaTimeRestart = dt;
		
		/*
		   Finally check to be sure that timestep will not overstep next plot output
		   time and adjust to match if necessary. This will adjust dt only if at the end
		   of the integration interval. In that case it will also recompute the Y[]
		   corresponding to the adjusted time interval.
		 */
		
		if (t + dt >= integrationData.t_max)
		{
			if (tid == 0)
			{
				/*
				   TODO
				   Copy back to CPU for dt_init next operator split integration.
				   Params2[2] = dt;
				*/
				
				dt = integrationData.t_max - t;
			}
			
			__syncthreads();
			
			updatePopulations(FplusSum, FminusSum, Y, Yzero, numberSpecies, dt);
		}
		
		__syncthreads();
		
		/* NOTE: eventually need to deal with special case Be8 <-> 2 He4. */
		
		/* Now that final dt is set, compute final sum of mass fractions sumX. */
		
		for (int i = tid; i < numberSpecies; i += blockDim.x)
		{
			/* Compute mass fraction X from abundance Y. */
			X[i] = massNum[i] * Y[i];
		}
		
		__syncthreads();
		sumX = NDreduceSum(X, numberSpecies);
		
		__syncthreads();
		
		if (tid == 0)
		{
			/* Increment the integration time and set the new timestep. */
			
			t += dt;
			timesteps++;
		}
		
		sumXLast = sumX;
	}
}


/* Device functions */

/*
   Determines whether an isotope specified by speciesIndex satisfies the
   asymptotic condition. Returns 1 if it does and 0 if not.
*/

__device__ inline bool checkAsy(fern_real Fminus, fern_real Y, fern_real dt)
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

__device__ inline fern_real asymptoticUpdate(fern_real Fplus, fern_real Fminus, fern_real Y, fern_real dt)
{
	/* Sophia He formula */
	return (Y + Fplus * dt) / (1.0 + Fminus * dt / Y);
}


/* Returns the Y specified by speciesIndex updated using the forward Euler method */

__device__ inline fern_real eulerUpdate(fern_real FplusSum, fern_real FminusSum, fern_real Y, fern_real dt)
{
	return Y + (FplusSum - FminusSum) * dt;
}

/*
   Performs a parallel sum reduction in O(log(length)) time

   The given array is overwritten by intermediate values during computation.
   The maximum array size is 2 * blockDim.x.
*/

__device__ fern_real reduceSum(fern_real *a, unsigned short length)
{
	const int tid = threadIdx.x;
	unsigned short k = length;
	
	do
	{
		k = (k + 1) / 2;
		
		if (tid < k && tid + k < length)
			a[tid] += a[tid + k];
		
		length = k;
		__syncthreads();
	}
	while (k > 1);
	
	return a[0];
}

/*
   Non-destructive sum reduction.
   Same as previous, but copies array to dsmem allocated
   to the global scratch_space before executing algorithm.
*/

__device__ fern_real NDreduceSum(fern_real *a, unsigned short length)
{
    const int tid = threadIdx.x;
    unsigned short k = length;
    fern_real *b;

    b = scratch_space;

    for (int i = tid; i < k; i += blockDim.x) {
        b[i] = a[i];
    }

    __syncthreads();

    do {
        k = (k + 1) / 2;

        if (tid < k && tid + k < length)
            b[tid] += b[tid + k];

        length = k;
        __syncthreads();
    } while (k > 1);

    return b[0];
}

/*
   Performs a parallel maximum reduction in O(log(length)) time

   The given array is overwritten by intermediate values during computation.
   The maximum array size is 2 * blockDim.x.
*/
__device__ fern_real reduceMax(fern_real *a, unsigned short length)
{
	const int tid = threadIdx.x;
	unsigned short k = length;
	
	do
	{
		k = (k + 1) / 2;
		
		if (tid < k && tid + k < length)
			#ifdef FERN_SINGLE
				a[tid] = fmaxf(a[tid], a[tid + k]);
			#else
				a[tid] = fmax(a[tid], a[tid + k]);
			#endif
		
		length = k;
		__syncthreads();
	}
	while (k > 1);
	
	return a[0];
}

/*
   Populates Fplus or Fminus
   Since the calculations for Fplus and Fminus are similar, the implementation
   of this function uses the term 'sign' to replace 'plus' and 'minus'.
*/

__device__ void populateF(fern_real *Fsign, fern_real *FsignFac, fern_real *Flux,
	unsigned short *MapFsign, unsigned short totalFsign)
{
	const int tid = threadIdx.x;
	
	for (int i = tid; i < totalFsign; i += blockDim.x)
	{
		Fsign[i] = FsignFac[i] * Flux[MapFsign[i]];
	}
}


/* Updates populations based on the trial timestep */

__device__ inline void updatePopulations(fern_real *FplusSum, fern_real *FminusSum,
	fern_real *Y, fern_real *Yzero, unsigned short numberSpecies, fern_real dt)
{
	const int tid = threadIdx.x;
	
	/* Parallel Update populations based on this trial timestep. */
	for (int i = tid; i < numberSpecies; i += blockDim.x)
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

__device__ void network_print(const Network &network)
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

size_t integrateNetwork_sharedSize(const Network &network)
{
	size_t size = 49100;

	return size;
}
