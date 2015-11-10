#ifndef Network_cuh
#define Network_cuh

#include "fern_math.h"
#include <vector>
#include <stdlib.h>

struct Network {
	unsigned short species = 0;
	unsigned short reactions = 0;
	unsigned int totalFplus = 0;
	unsigned int totalFminus = 0;

	// Network data

	unsigned char *Z = NULL; // [species]
	unsigned char *N = NULL; // [species]
	char **isotopeLabel = NULL;
	char **reactionLabel = NULL;

	std::vector<fern_real> FplusFac; // [totalFplus]
	std::vector<fern_real> FminusFac; // [totalFminus]

	std::vector<unsigned short> MapFplus; // [totalFplus]
	std::vector<unsigned short> MapFminus; // [totalFminus]

	unsigned short *FplusMax = NULL; // [species]
	unsigned short *FminusMax = NULL; // [species]

	fern_real massTol = 0.0;
	fern_real fluxFrac = 0.0;

	// Reaction data

	fern_real *P[7]; // [reactions]
	unsigned char *numReactingSpecies = NULL; // [reactions]

	fern_real *statFac = NULL; // [reactions]
	fern_real *Q = NULL; // [reactions]
	int *reactant[3]; // [reactions]
	int *product[3]; // [reactions]

	//Partial Equilibrium
	int *RGclassByRG = NULL; //[numRG]
	int *RGmemberIndex = NULL; // [reactions]
	int *PEnumProducts = NULL; // [reactions]
	int *isReverseR = NULL; //[reactions]
	int *ReacParent = NULL; // [reactions]
	int numRG = 0;
	int *pEquilbyRG = NULL; //[numRG]
	int *pEquilbyReac = NULL; //[reactions]
	int *ReacRG = NULL; //[reactions]
	int *RGParent = NULL; //[numRG]
	int *RGclass = NULL; // [reactions]
	int **reacVector = NULL; // [reactions][species]

	Network() {};

	/**	Reads the network data file line by line

	 This file is expected to have 4 lines per isotope with the line structure
	 isotopeSymbol A  Z  N  Y  MassExcess
	 pf00 pf01 pf02 pf03 pf04 pf05 pf06 pf07
	 pf10 pf11 pf12 pf13 pf14 pf15 pf16 pf17
	 pf20 pf21 pf22 pf23 pf24 pf25 pf26 pf27
	 where isotopeSymbol is an isotope label, A=Z+N is the atomic mass number, Z is the proton number,
	 N is the neutron number, Y is the current abundance, MassExcess is the mass
	 excess in MeV, and the pf are 24 values of the partition function for that isotope at
	 different values of the temperature that will form a table for interpolation in temperature.
	 The assumed 24 values of the temperature for the partition function table are
	 { 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
	 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 } in units of 10^9 K.
	 All fields on a line are separated by a blank space and there is no whitespace in the isotopeSymbol.
	 The type signature of these four lines corresponding to a single isotope is
	 string int int int double double
	 double double double double double double double double
	 double double double double double double double double
	 double double double double double double double double
	 Here is an example for two isotopes:

	 ca40 40 20 20 0.0 -34.846
	 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
	 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
	 1.0 1.0 1.0 1.01 1.04 1.09 1.2 1.38
	 ti44 44 22 22 0.0 -37.548
	 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
	 1.0 1.0 1.0 1.0 1.01 1.03 1.08 1.14
	 1.23 1.35 1.49 1.85 2.35 3.01 3.86 4.94

	 A file with this format is written from the Java code to the file
	 output/CUDAnetwork.inp using the Java stream toCUDAnet.
	 */
	void loadNetwork(const char *filename);

	/**	Reads rate parameter data file line by line

	 The file is expected to have one reaction per line with the line structure
	 p0 p1 p2 p3 p4 p5 p6 reactionLabel
	 where the pn are the values of the 7 Reaclib parameters for a reaction,
	 reactionLabel is a label for the reaction that must contain no whitespace, and
	 all fields on a line are separated by a space character.
	 */
	void loadReactions(const char *filename);

	// Define vec_4i as the type int[4]
	// This is bad and should probably be changed during
	// a revision to parseFlux().
	typedef int vec_4i[4];

	/**	Finds the contributions to F+ and F- of each reaction for each isotope

	 This should be executed only once at the beginning of the entire
	 calculation to determine the structure of the network.
	 */
	void parseFlux(int *numProducts, vec_4i *reactantZ, vec_4i *reactantN,
			vec_4i *productZ, vec_4i *productN);

	/**	Allocates vectors on the host

	 Assumes that `networks` and `reactions` are set to the desired
	 size
	 */
	void allocate();
	void setSizes(const Network &source);

	void print();
};

#endif
