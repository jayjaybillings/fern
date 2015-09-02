
#include "Network.cuh"
#include <stdio.h>
#include <assert.h>
#include "FERNIntegrator.cuh"


Network::Network()
{
}


void Network::loadNetwork(const char *filename)
{
	// Unused variables
	char isotopeLabel[10];
	unsigned short A;
	fern_real massExcess;
	fern_real pf;
	fern_real Y;
	
	
	FILE *file = fopen(filename, "r");
	
	// Exit if the file doesn't exist or can't be read
	
	if (!file)
	{
		fprintf(stderr, "Could not read file '%s'\n", filename);
		exit(1);
	}
	
	// Read 4 lines at a time
	
	for (int n = 0; n < species; n++)
	{
		int status;
		
		// Line #1
		
		#ifdef FERN_SINGLE
			status = fscanf(file, "%s %hu %hhu %hhu %f %f\n",
				isotopeLabel, &A, &Z[n], &N[n], &Y, &massExcess);
		#else
			status = fscanf(file, "%s %hu %hhu %hhu %lf %lf\n",
				isotopeLabel, &A, &Z[n], &N[n], &Y, &massExcess);
		#endif
		
		if (status == EOF)
			break;
		
		// Line #2...4
		
		for (int i = 0; i < 8 * 3; i++)
		{
			#ifdef FERN_SINGLE
				status = fscanf(file, "%f", &pf);
			#else
				status = fscanf(file, "%lf", &pf);
			#endif
		}
	}
}


void Network::loadReactions(const char *filename)
{
	static const bool displayInput = false;
	
	// Unused variables
	char reactionLabel[100];
	int RGclass;
	int RGmemberIndex;
	int reaclibClass;
	int isEC;
	int isReverseR;
	int ProductIndex[4];
	
	// Allocate the host-only memory to be used by parseFlux()
	int *numProducts = new int [reactions];
	
	// Each element of these dynamic arrays are pointers to static arrays of size 4.
	vec_4i *reactantZ = new vec_4i [reactions]; // [reactions]
	vec_4i *reactantN = new vec_4i [reactions]; // [reactions]
	vec_4i *productZ = new vec_4i [reactions]; // [reactions]
	vec_4i *productN = new vec_4i [reactions]; // [reactions]
	
	
	FILE *file = fopen(filename, "r");
	
	// Exit if the file doesn't exist or can't be read
	
	if (!file)
	{
		fprintf(stderr, "File Input Error: No readable file named %s\n", filename);
		exit(1);
	}
	
	// Read eight lines at a time
	
	for (int n = 0; n < reactions; n++)
	{
		int status;
		
		// Line #1

		#ifdef FERN_SINGLE		
			status = fscanf(file, "%s %d %d %d %hhu %d %d %d %f %f",
				reactionLabel, &RGclass, &RGmemberIndex, &reaclibClass,
				&numReactingSpecies[n], &numProducts[n], &isEC, &isReverseR,
				&statFac[n], &Q[n]);
		#else
			status = fscanf(file, "%s %d %d %d %hhu %d %d %d %lf %lf",
				reactionLabel, &RGclass, &RGmemberIndex, &reaclibClass,
				&numReactingSpecies[n], &numProducts[n], &isEC, &isReverseR,
				&statFac[n], &Q[n]);
		#endif
		
		if (status == EOF)
			break;
		
		if (displayInput)
		{
			printf("Reaction Index = %d\n", n);
			printf("isReverseR = %d reaclibIndex = %d\n",
				isReverseR, reaclibClass);
			printf("%s %d %d %d %d %d %d %d %f %f\n",
				reactionLabel, RGclass, RGmemberIndex, reaclibClass,
				numReactingSpecies[n], numProducts[n], isEC,
				isReverseR, statFac[n], Q[n]);
		}
		
		// Line #2
		
		if (displayInput)
			printf("P: { ");
		
		for (int i = 0; i < 7; i++)
		{
			#ifdef FERN_SINGLE
				status = fscanf(file, "%f", &P[i][n]);
			#else
				status = fscanf(file, "%lf", &P[i][n]);
			#endif
			
			if (displayInput)
				printf("%f, ", P[i][n]);
		}
		
		if (displayInput)
			printf("}\n");
		
		// Line #3
		
		for (int mm = 0; mm < numReactingSpecies[n]; mm++)
		{
			status = fscanf(file, "%d", &reactantZ[n][mm]);
			
			if (displayInput)
				printf("\tReactant[%d]: Z=%d\n", mm, reactantZ[n][mm]);
		}
		
		// Line #4
		
		for (int mm = 0; mm < numReactingSpecies[n]; mm++)
		{
			status = fscanf(file, "%d", &reactantN[n][mm]);
			
			if (displayInput)
				printf("\tReactant[%d]: N=%d\n", mm, reactantN[n][mm]);
		}
		
		// Line #5
		
		for (int mm = 0; mm < numProducts[n]; mm++)
		{
			status = fscanf(file, "%d", &productZ[n][mm]);
			
			if (displayInput)
				printf("\tProduct[%d]: Z=%d\n", mm, productZ[n][mm]);
		}
		
		// Line #6
		
		for (int mm = 0; mm < numProducts[n]; mm++)
		{
			status = fscanf(file, "%d", &productN[n][mm]);
			
			if (displayInput)
				printf("\tProduct[%d]: N=%d\n", mm, productN[n][mm]);
		}
		
		// Line #7
		
		for (int mm = 0; mm < numReactingSpecies[n]; mm++)
		{
			status = fscanf(file, "%hu", &reactant[mm][n]);
			
			if (displayInput)
				printf("\treactant[%d]: N=%d\n", mm, reactant[mm][n]);
		}
		
		// Line #8
		
		for (int mm = 0; mm < numProducts[n]; mm++)
		{
			status = fscanf(file, "%d", &ProductIndex[mm]);
			
			if (displayInput)
				printf("\tProductIndex[%d]: N=%d\n", mm, ProductIndex[mm]);
		}
		
		if (displayInput)
			printf("\n");
	}
	
	fclose(file);
	
	
	// We're not done yet.
	// Finally parse the flux
	
	parseFlux(numProducts, reactantZ, reactantN, productZ, productN);
	
	
	// Cleanup dynamic memory
	
	delete [] numProducts;
	delete [] reactantZ;
	delete [] reactantN;
	delete [] productZ;
	delete [] productN;
}


void Network::parseFlux(int *numProducts, vec_4i *reactantZ, vec_4i *reactantN,
	vec_4i *productZ, vec_4i *productN)
{
	const static bool showParsing = false;
	
	// These tempInt blocks will become MapFPlus and MapFMinus eventually.
	size_t tempIntSize = species * reactions / 2;
	unsigned short *tempInt1 = new unsigned short [tempIntSize];
	unsigned short *tempInt2 = new unsigned short [tempIntSize];
	
	// Access elements by reacMask[speciesIndex + species * reactionIndex].
	int *reacMask = new int [species * reactions]; // [species][reactions]
	
	int *numFluxPlus = new int [species];
	int *numFluxMinus = new int [species];
	
	// Start of Guidry's original parseF() code
	
	if (showParsing)
		printf("Use parseF() to find F+ and F- flux components for each species:\n");
	
	int incrementPlus = 0;
	int incrementMinus = 0;
	
	totalFplus = 0;
	totalFminus = 0;
	
	// Loop over all isotopes in the network
	for (int i = 0; i < species; i++)
	{
		int total = 0;
		int numFplus = 0;
		int numFminus = 0;
		
		// Loop over all possible reactions for this isotope, finding those that
		// change its population up (contributing to F+) or down (contributing
		// to F-).
		
		for (int j = 0; j < reactions; j++)
		{
			int totalL = 0;
			int totalR = 0;
			
			// Loop over reactants for this reaction
			for (int k = 0; k < numReactingSpecies[j]; k++)
			{
				if (Z[i] == reactantZ[j][k] && N[i] == reactantN[j][k])
					totalL++;
			}
			
			// Loop over products for this reaction
			for (int k = 0; k < numProducts[j]; k++)
			{
				if (Z[i] == productZ[j][k] && N[i] == productN[j][k])
					totalR++;
			}
			
			total = totalL - totalR;
			
			if (total > 0)       // Contributes to F- for this isotope
			{
				numFminus++;
				reacMask[i + species * j] = -total;
				tempInt2[incrementMinus + numFminus - 1] = j;
				// if (showParsing)
				// 	printf("%s reacIndex=%d %s nReac=%d nProd=%d totL=%d totR=%d tot=%d F-\n",
				// 		   isoLabel[i], j, reacLabel[j], NumReactingSpecies[j], NumProducts[j], totalL,
				// 		   totalR, total);
			}
			else if (total < 0)  // Contributes to F+ for this isotope
			{
				numFplus++;
				reacMask[i + species * j] = -total;
				tempInt1[incrementPlus + numFplus - 1] = j;
				// if (showParsing)
				// 	printf("%s reacIndex=%d %s nReac=%d nProd=%d totL=%d totR=%d tot=%d F+\n",
				// 		   isoLabel[i], j, reacLabel[j], NumReactingSpecies[j], NumProducts[j], totalL,
				// 		   totalR, total);
			}
			else                 // Does not contribute to flux for this isotope
			{
				reacMask[i + species * j] = 0;
			}
		}
		
		// Keep track of the total number of F+ and F- terms in the network for all isotopes
		totalFplus += numFplus;
		totalFminus += numFminus;
		
		numFluxPlus[i] = numFplus;
		numFluxMinus[i] = numFminus;
		
		incrementPlus += numFplus;
		incrementMinus += numFminus;
		
		// if (showParsing == 1)
		// 	printf("%d %s numF+ = %d numF- = %d\n", i, isoLabel[i], numFplus, numFminus);
	}
	
	// Display some cases
	
	printf("\n");
	printf("PART OF FLUX-ISOTOPE COMPONENT ARRAY (-n --> F-; +n --> F+ for given isotope):\n");
	
	printf("\n");
	printf("FLUX SPARSENESS: Non-zero F+ = %d; Non-zero F- = %d, out of %d x %d = %d possibilities.\n",
		totalFplus, totalFminus, reactions, species, reactions * species);
	
	
	
	/*******************************************/
	
	
	// Create 1D arrays to hold non-zero F+ and F- for all reactions for all isotopes,
	// the arrays holding the species factors FplusFac and FminusFac,
	// and also arrays to hold their sums for each isotope. Note that parseF() must
	// be run first because it determines totalFplus and totalFminus.
	
	FplusFac = new fern_real [totalFplus];
	FminusFac = new fern_real [totalFminus];
	
	// Create 1D arrays that will hold the index of the isotope for the F+ or F- term
	MapFplus = new unsigned short [totalFplus];
	MapFminus = new unsigned short [totalFminus];
	
	// Create 1D arrays that will be used to map finite F+ and F- to the Flux array.
	
	int *FplusIsotopeCut = new int [species];
	int *FminusIsotopeCut = new int [species];
	
	int *FplusIsotopeIndex = new int [totalFplus];
	int *FminusIsotopeIndex = new int [totalFminus];
	
	
	
	FplusIsotopeCut[0] = numFluxPlus[0];
	FminusIsotopeCut[0] = numFluxMinus[0];
	for (int i = 1; i < species; i++)
	{
		FplusIsotopeCut[i] = numFluxPlus[i] + FplusIsotopeCut[i - 1];
		FminusIsotopeCut[i] = numFluxMinus[i] + FminusIsotopeCut[i - 1];
	}
	
	int currentIso = 0;
	for (int i = 0; i < totalFplus; i++)
	{
		FplusIsotopeIndex[i] = currentIso;
		if (i == (FplusIsotopeCut[currentIso] - 1)) currentIso ++;
	}
	
	currentIso = 0;
	for (int i = 0; i < totalFminus; i++)
	{
		FminusIsotopeIndex[i] = currentIso;
		if (i == (FminusIsotopeCut[currentIso] - 1)) currentIso ++;
	}
	
	// Diagnostic output
	// if (showFparsing == 1)
	// {
	// 	printf("\n\n");
	// 	printf("MAX F+ and F- INDEX FOR EACH ISOTOPE:\n");
	// 	for (int i = 0; i < species; i++)
	// 	{
	// 		printf("\n");
	// 		printf("Isotope index = %d  %s  Max index F+ = %d  Max index F- = %d\n",
	// 			   i, isoLabel[i], FplusIsotopeCut[i] - 1, FminusIsotopeCut[i] - 1);
	// 	}
	// }
	
	for (int i = 0; i < totalFplus; i++)
	{
		MapFplus[i] = tempInt1[i];
	}
	
	for (int i = 0; i < totalFminus; i++)
	{
		MapFminus[i] = tempInt2[i];
	}
	
	// Populate the FplusMin and FplusMax arrays
	unsigned short *FplusMin = new unsigned short [species];
	unsigned short *FminusMin = new unsigned short [species];
	
	
	FplusMin[0] = 0;
	FplusMax[0] = numFluxPlus[0] - 1;
	for (int i = 1; i < species; i++)
	{
		FplusMin[i] = FplusMax[i - 1] + 1;
		FplusMax[i] = FplusMin[i] + numFluxPlus[i] - 1 ;
	}
	// Populate the FminusMin and FminusMax arrays
	FminusMin[0] = 0;
	FminusMax[0] = numFluxMinus[0] - 1;
	for (int i = 1; i < species; i++)
	{
		FminusMin[i] = FminusMax[i - 1] + 1;
		FminusMax[i] = FminusMin[i] + numFluxMinus[i] - 1 ;
	}
	
	// Populate the FplusFac and FminusFac arrays that hold the factors counting the
	// number of occurences of the species in the reaction.  Note that this can only
	// be done after parseF() has been run to give reacMask[i][j].
	
	int tempCountPlus = 0;
	int tempCountMinus = 0;
	for (int i = 0; i < species; i++)
	{
		for (int j = 0; j < reactions; j++)
		{
			if (reacMask[i + species * j] > 0)
			{
				FplusFac[tempCountPlus] = (fern_real)reacMask[i + species * j];
				tempCountPlus++;
			}
			else if (reacMask[i + species * j] < 0)
			{
				FminusFac[tempCountMinus] = -(fern_real) reacMask[i + species * j];
				tempCountMinus++;
			}
		}
	}
	
	
	// Clean up dynamic memory
	delete [] reacMask;
	
	delete [] FplusIsotopeCut;
	delete [] FminusIsotopeCut;
	
	delete [] FplusIsotopeIndex;
	delete [] FminusIsotopeIndex;
	
	delete [] tempInt1;
	delete [] tempInt2;
	
	delete [] numFluxPlus;
	delete [] numFluxMinus;
	
	delete [] FplusMin;
	delete [] FminusMin;
}


void Network::allocate()
{
	// Allocate the network data
	
	Z = new unsigned char[species];
	N = new unsigned char[species];
	
	FplusMax = new unsigned short [species];
	FminusMax = new unsigned short [species];
	
	
	// Allocate the reaction data
	
	for (int i = 0; i < 7; i++)
		P[i] = new fern_real[reactions];
	
	numReactingSpecies = new unsigned char[reactions];
	statFac = new fern_real[reactions];
	Q = new fern_real[reactions];
	
	for (int i = 0; i < 3; i++)
		reactant[i] = new unsigned short[reactions];
}


void Network::cudaAllocate()
{
	// Allocate network data
	
	cudaMalloc(&Z, sizeof(unsigned char) * species);
	cudaMalloc(&N, sizeof(unsigned char) * species);
	
	cudaMalloc(&FplusFac, sizeof(fern_real) * totalFplus);
	cudaMalloc(&FminusFac, sizeof(fern_real) * totalFminus);
	cudaMalloc(&MapFplus, sizeof(unsigned short) * totalFplus);
	cudaMalloc(&MapFminus, sizeof(unsigned short) * totalFminus);
	
	cudaMalloc(&FplusMax, sizeof(unsigned short) * species);
	cudaMalloc(&FminusMax, sizeof(unsigned short) * species);
	
	// Allocate reaction data
	
	for (int i = 0; i < 7; i++)
	{
		cudaMalloc(&P[i], sizeof(fern_real) * reactions);
	}
	
	cudaMalloc(&numReactingSpecies, sizeof(unsigned char) * reactions);
	cudaMalloc(&statFac, sizeof(fern_real) * reactions);
	cudaMalloc(&Q, sizeof(fern_real) * reactions);
	
	for (int i = 0; i < 3; i++)
		cudaMalloc(&reactant[i], sizeof(unsigned short) * reactions);
}


void Network::setSizes(const Network &source)
{
	species = source.species;
	reactions = source.reactions;
	totalFplus = source.totalFplus;
	totalFminus = source.totalFminus;
}


void Network::cudaCopy(const Network &source, cudaMemcpyKind kind)
{
	// Copy scalars
	
	massTol = source.massTol;
	fluxFrac = source.fluxFrac;
	
	// Copy network vectors
	
	cudaMemcpy(Z, source.Z, sizeof(unsigned char) * species, kind);
	cudaMemcpy(N, source.N, sizeof(unsigned char) * species, kind);
	
	FERNIntegrator::checkCudaErrors();
	cudaMemcpy(FplusFac, source.FplusFac, sizeof(fern_real) * totalFplus, kind);
	FERNIntegrator::checkCudaErrors();
	cudaMemcpy(FminusFac, source.FminusFac, sizeof(fern_real) * totalFminus, kind);
	FERNIntegrator::checkCudaErrors();
	cudaMemcpy(MapFplus, source.MapFplus, sizeof(unsigned short) * totalFplus, kind);
	cudaMemcpy(MapFminus, source.MapFminus, sizeof(unsigned short) * totalFminus, kind);
	
	cudaMemcpy(FplusMax, source.FplusMax, sizeof(unsigned short) * species, kind);
	cudaMemcpy(FminusMax, source.FminusMax, sizeof(unsigned short) * species, kind);
	
	// Copy reaction vectors
	
	
	for (int i = 0; i < 7; i++)
	{
		cudaMemcpy(P[i], source.P[i], sizeof(fern_real) * reactions, kind);
	}
	
	cudaMemcpy(numReactingSpecies, source.numReactingSpecies,
		sizeof(unsigned char) * reactions, kind);
	cudaMemcpy(statFac, source.statFac, sizeof(fern_real) * reactions, kind);
	cudaMemcpy(Q, source.Q, sizeof(fern_real) * reactions, kind);
	
	for (int i = 0; i < 3; i++)
	{
		cudaMemcpy(reactant[i], source.reactant[i],
			sizeof(unsigned short) * reactions, kind);
	}
}


void Network::print()
{
	// Network data
	
	printf("species: %d\n", species);
	
	printf("Z: { ");
	for (int i = 0; i < species; i++)
		printf("%4d ", Z[i]);
	printf("}\n");
	
	printf("N: { ");
	for (int i = 0; i < species; i++)
		printf("%4d ", N[i]);
	printf("}\n");
	
	// Reaction data
	
	printf("\n");
	
	printf("reactions: %d\n", reactions);
	
	for (int n = 0; n < 7; n++)
	{
		printf("P[%d]: { ", n);
		for (int i = 0; i < reactions; i++)
			printf("%e ", P[n][i]);;
		printf("\n");
	}
	
	printf("numReactingSpecies: { ");
	for (int i = 0; i < reactions; i++)
		printf("%4d ", numReactingSpecies[i]);
	printf("}\n");
	
	
	printf("statFac: { ");
	for (int i = 0; i < reactions; i++)
		printf("%e ", statFac[i]);
	printf("}\n");
	
	
	printf("Q: { ");
	for (int i = 0; i < reactions; i++)
		printf("%e ", Q[i]);
	printf("}\n");
	
	for (int n = 0; n < 3; n++)
	{
		printf("reactant[%d]: { ", n);
		for (int i = 0; i < reactions; i++)
			printf("%4d ", reactant[n][i]);
		printf("}\n");
	}
	
	printf("totalFplus: %d\n", totalFplus);
	printf("totalFminus: %d\n", totalFminus);
	
	printf("FplusFac: { ");
	for (int i = 0; i < totalFplus; i++)
		printf("%e ", FplusFac[i]);
	printf("}\n");
	
	printf("FminusFac: { ");
	for (int i = 0; i < totalFminus; i++)
		printf("%e ", FminusFac[i]);
	printf("}\n");
	
	printf("MapFplus: { ");
	for (int i = 0; i < totalFplus; i++)
		printf("%4u ", MapFplus[i]);
	printf("}\n");
	
	printf("MapFminus: { ");
	for (int i = 0; i < totalFminus; i++)
		printf("%4u ", MapFminus[i]);
	printf("}\n");
	
	printf("FplusMax: { ");
	for (int i = 0; i < species; i++)
		printf("%4u ", FplusMax[i]);
	printf("}\n");
	
	printf("FminusMax: { ");
	for (int i = 0; i < species; i++)
		printf("%4u ", FminusMax[i]);
	printf("}\n");
}
