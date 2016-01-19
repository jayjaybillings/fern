#include "Network.hpp"
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include "FERNIntegrator.hpp"


Network::Network()
{
}


void Network::loadNetwork(const char *filename)
{
	// Unused variables
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
	speciesLabel = (char **) malloc(sizeof(char *) * species);
	speciesFamily = (char **) malloc(sizeof(char *) * species);
  //boolean for plotting
const bool plotOutput = 0;
const bool plotRadicals = 0;
  if(plotOutput == 1) {  
    printf("BEGIN PLOTTING (COLHEADER)\n");
    printf("t ");
  }
	for (int n = 0; n < species; n++)
	{
		speciesLabel[n] = (char *) malloc(sizeof(char) * 10);
		speciesFamily[n] = (char *) malloc(sizeof(char) * 10);
		int status;
		
		// Line #1
		
		#ifdef FERN_SINGLE
			status = fscanf(file, "%s %s %hu %hhu %hhu %f %f\n",
				speciesLabel[n], speciesFamily[n], &A, &Z[n], &N[n], &Y, &massExcess);
		#else
			status = fscanf(file, "%s %s %hu %hhu %hhu %lf %lf\n",
				speciesLabel[n], speciesFamily[n], &A, &Z[n], &N[n], &Y, &massExcess);
		#endif
      //printf("Y[%d]: %s %e %d %d\n", n, speciesLabel[n], Y, Z[n], N[n]);
		if(status == EOF)
			break;
    
    if(plotRadicals == 0 && plotOutput == 1) {
      //plot long lived non-radicals 
      if(n == 0 || n == 12 || n == 6 || n == 13 || n==14 || n==18 || n==27 || n==101)
        printf("%s ", speciesLabel[n]);
    } else if (plotRadicals == 1 && plotOutput == 1) {
      //plot short lived radicals 
      if(n == 75 || n == 3 || n == 76 || n == 77)
        printf("%s ", speciesLabel[n]);
    }

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
  if(plotOutput == 1) {
    printf("\nEND PLOTTING (COLHEADER)\n");
  }
}


void Network::loadReactions(const char *filename)
{
	static const bool displayInput = false;
	static const bool displayPEInput = false;
	
	// Unused variables
	int reaclibClass;
	int isEC;
	
	// Allocate the host-only memory to be used by parseFlux()
	int *numProducts = new int [reactions];
  int *RGclass = new int [reactions];
	
	// Each element of these dynamic arrays are pointers to static arrays of size 4.
	vec_4i *reactantC = new vec_4i [reactions]; // [reactions]
	vec_4i *reactantN = new vec_4i [reactions]; // [reactions]
	vec_4i *productC = new vec_4i [reactions]; // [reactions]
	vec_4i *productN = new vec_4i [reactions]; // [reactions]
	
	
	FILE *file = fopen(filename, "r");
	
	// Exit if the file doesn't exist or can't be read
	
	if (!file)
	{
		fprintf(stderr, "File Input Error: No readable file named %s\n", filename);
		exit(1);
	}
	
	// Read eight lines at a time
  numRG = 0;
  reactionLabel = (char **) malloc(sizeof(char *) * reactions);
  photoID = (char **) malloc(sizeof(char *) * reactions);
  reacVector = (int **) malloc(sizeof(int *) * reactions);
	int RGParent = 0;
	for (int n = 0; n < reactions; n++)
	{
		reactionLabel[n] = (char *) malloc(sizeof(char) * 150);
		photoID[n] = (char *) malloc(sizeof(char) * 50);
		reacVector[n] = (int *) malloc(sizeof(int) * species);
		int status;
		
		// Line #1

		#ifdef FERN_SINGLE		
			status = fscanf(file, "%s %s %d %d %d %hhu %d %d %d %d %f %f",
				reactionLabel[n], photoID[n], &RGclass[n], &RGmemberIndex[n], &reaclibClass,
				&numReactingSpecies[n], &numProducts[n], &isEC, &isReverseR[n],
				&reacType[n], &statFac[n], &Q[n]);
		#else
			status = fscanf(file, "%s %s %d %d %d %hhu %d %d %d %d %lf %lf",
				reactionLabel[n], photoID[n], &RGclass[n], &RGmemberIndex[n], &reaclibClass,
				&numReactingSpecies[n], &numProducts[n], &isEC, &isReverseR[n],
				&reacType[n], &statFac[n], &Q[n]);
		#endif
//	printf("Reaction[%d] is %s\n", n, reactionLabel[n]);
	
		if (status == EOF)
			break;
		
		if (displayInput)
		{
			printf("Reaction Index = %d\n", n);
			printf("isReverseR = %d reaclibIndex = %d\n",
				isReverseR[n], reaclibClass);
			printf("%s %s %d %d %d %d %d %d %d %d %f %f\n",
				reactionLabel[n], photoID[n], RGclass[n], RGmemberIndex[n], reaclibClass,
				numReactingSpecies[n], numProducts[n], isEC,
				isReverseR[n], reacType[n], statFac[n], Q[n]);
		}
		// Line #2
		
		if (displayInput)
			printf("P: { ");
		
		for (int i = 0; i < 19; i++)
		{
			#ifdef FERN_SINGLE
				status = fscanf(file, "%e", &P[i][n]);
				printf("%e, ", P[i][n]);
			#else
				status = fscanf(file, "%le", &P[i][n]);
			#endif
			
			if (displayInput)
				printf("%e, ", P[i][n]);
		}
		
		if (displayInput)
			printf("}\n");
		
		// Line #3
    //scan the reactant coefficients
		for (int mm = 0; mm < numReactingSpecies[n]; mm++)
		{
			status = fscanf(file, "%f", &reactantC[n][mm]);
			
			if (displayInput)
				printf("\tReactant[%d]: Coeff=%f\n", mm, reactantC[n][mm]);
		}
		
		// Line #4
    //scan the product coefficients which will alter the generated abundances from this reaction.
		for (int mm = 0; mm < numProducts[n]; mm++)
		{
			status = fscanf(file, "%f", &productC[n][mm]);
			
			if (displayInput)
				printf("\tProduct[%d]: Coeff=%f\n", mm, productC[n][mm]);
		}
		
		// Line #5
		
		for (int mm = 0; mm < numReactingSpecies[n]; mm++)
		{
			status = fscanf(file, "%d", &reactant[mm][n]);
      //"subtract" reactants from reacVector (PE)
      for(int i = 0; i < species; i++) {
        if(i==reactant[mm][n]){
          reacVector[n][i]--;
        }      
      }
			if (displayInput)
				printf("\treactant[%d]: N=%d\n", mm, reactant[mm][n]);
		}
		
		// Line #6
		
		for (int mm = 0; mm < numProducts[n]; mm++)
		{
			status = fscanf(file, "%d", &product[mm][n]);
      //"add" products to reacVector (PE)
      for(int i = 0; i < species; i++) {
        if(i==product[mm][n]){
          reacVector[n][i]++;
        }      
      }
			
			if (displayInput)
				printf("\tProductIndex[%d]: N=%d\n", mm, product[mm][n]);
		}
    PEnumProducts[n] = numProducts[n];
	}
  //PartialEquilibrium: define Reaction Groups based on ReacVector
  //reworked so parsing to reaction groups does not depend on the reaction input
  //file having RG members in order
  int isRGmember = 0;
  int RGmemberID = 0;
  //each reaction's home RGid
  int *reactionRG = new int [reactions];
  int *reacPlaced = new int [reactions];
  for(int p = 0; p < reactions; p++) {
    reacPlaced[p] = 0;
  }
  //member ID within this reaction's RG
  int *RGmemID = new int [reactions];
  int *RGnumMembers = new int [numRG];
  for(int j = 0; j < reactions; j++) {
    //if this reaction doesn't yet have an RG home of its own...
    if(reacPlaced[j] == 0) {
      RGmemberID = 0;
      RGmemID[j] = RGmemberID;
      ReacParent[j] = j;
      if(displayPEInput) {
	      printf("\nRG #%d, Parent Reaction: %s, photoID: %s\n", numRG, reactionLabel[ReacParent[j]], photoID[j]);
	    	printf("-----\n");
        printf("Reaction %s ID[%d], RGmemID: %d\nReaction Vector: ", reactionLabel[j], j, RGmemID[j]);
        for (int q = 0; q < species; q++) {
          printf("%d ", reacVector[j][q]);
        }
	    	printf("\n");
      }
      for(int n = 0; n < reactions; n++) {
        for (int i = 0; i < species; i++) {
          //if reaction j has a different species than reaction n, check n+1
          //also if n = j, skip it
          if(abs(reacVector[j][i]) != abs(reacVector[n][i]) || j == n) {
            isRGmember = 0;
            break;
          } else {
            isRGmember = 1;
          }
        }
          //if reac n was determined to have same species as reac j, 
        if(isRGmember == 1) {
          //give reac n the current reaction group ID. 
          reacPlaced[n] = 1;
          RGmemberID++;
          reactionRG[n] = numRG; 
          RGmemID[n] = RGmemberID;
          ReacParent[n] = j;
          RGid[numRG] = j;
          if(displayPEInput) {
  	        printf("ReacParent: %d,  RGmemID: %d\n", ReacParent[n], RGmemID[n]);
            printf("Reaction %s ID[%d]\nReaction Vector: ", reactionLabel[n], n);
            for (int q = 0; q < species; q++) {
              printf("%d ", reacVector[n][q]);
            }
	      		printf("\n");
          }
        }
      }
    //indicates number of members in each reaction group
    RGnumMembers[numRG] = RGmemberID + 1;
    if(displayPEInput)
      printf("Number Members: %d\n", RGnumMembers[numRG]);
    numRG++;
    }
  }
//  printf("numRG: %d", numRG);
	fclose(file); 
	
	
	// We're not done yet.
	// Finally parse the flux
	
	parseFlux(numProducts, reactant, reactantC, product, productC);
	
	
	// Cleanup dynamic memory
	
	delete [] numProducts;
	delete [] reactantC;
	delete [] reactantN;
	delete [] productC;
	delete [] productN;
} //end loadReactions

void Network::loadPhotolytic(const char *filename) {
	static const bool displayPhotoInput = false;
	FILE *file = fopen(filename, "r");
	
	// Exit if the file doesn't exist or can't be read
	
	if (!file)
	{
		fprintf(stderr, "File Input Error: No readable file named %s\n", filename);
		exit(1);
	}
  photoLabel = (char **) malloc(sizeof(char *) * reactions);
  pparamID = (char **) malloc(sizeof(char *) * reactions);
  for(int t = 0; t < photolytic; t++) {
    paramNumID[0][t] = -1;
    paramNumID[1][t] = -1;
  }
	for (int n = 0; n < photoparams; n++) {
		photoLabel[n] = (char *) malloc(sizeof(char) * 100);
		pparamID[n] = (char *) malloc(sizeof(char) * 50);
		int status;
    int setParam = 1;
		
		// Line #1

		status = fscanf(file, "%s %s", photoLabel[n], pparamID[n]);
		
		if (status == EOF)
			break;
	//		printf(" photoLab = %s ppram= %s\n",	photoLabel[n], pparamID[n]);
  // Param Lines
  // for this polynomial coefficient (a_0,a_1, ... ,a_j)...
    int aparamCount = -1;
    for (int j = 0; j < 49; j++) {
      if(j % 7 == 0) {
        aparamCount++;
      }
      if(displayPhotoInput)
        printf("params for a%d, j: %d= ", aparamCount, j);
      // ...save parameter for 7 interpolation altitudes for each coefficient.
			  #ifdef FERN_SINGLE
				  status = fscanf(file, "%e", &aparam[j][n]);
          if(displayPhotoInput)
    				printf("%e, ", aparam[j][n]);
	  		#else
		  		status = fscanf(file, "%le", &aparam[j][n]);
          if(displayPhotoInput)
			  	  printf("%le, ", aparam[j][n]);
  			#endif
      if(displayPhotoInput)
        printf("\n");
  	}

      //check parameter length for fast comparison with the photolytic reaction ID
      int paramLength = 0;
      while(pparamID[n][paramLength] != '\0') {
        paramLength++;
      }
    //connect photolysis params to corresponding reactions from loadReactions function
    //based on pparamID, eg "fjo3a"
    //currently set up for a maximum of TWO parameters with TWO multipliers (ie, 0.1*fjno3a+0.9*fjno3b)
    for(int i = 0; i < photolytic; i++) {
      paramMult[0][i] = 1; //setting default first multiplier to 1
      paramMult[1][i] = 0; //setting default second multiplier to 0
      //set default photolytic parameter IDs to -1 so 0 is an applicable ID.
      //-1 for both will imply that this photolytic reaction does not have a 
      //corresponding parameter, and its reaction rate should be set to zero. 
      //if the second remains zero, this reaction only requires one 
      //photolytic parameter.
      //effect of strlen without the need for <string> library
      int length = 0;
      //photoID is the full string of the photolytic parameter name such as 0.1*fjno3a+0.9*fjno3b or fjo3a which is held in the main rate library
      while(photoID[i][length] != '\0') {
        length++;
      }

      //if length > 1, this reaction has at least one corresponding photolytic parameter  
      if(length > 1) {
        //check if photoID has a "+" in it, implying a combination of photolytic parameters (ie. fjno3a+fjno3b)
        for(int j = 0; j < length; j++) {
          if(photoID[i][j] == 43) {
//            printf("This Reaction has a plus: %s\n", photoID[i]);
            //43 is the ascii code for "+"
            //now j = location of "+"
            char firstID[j]; //the chunk before "+"
            char secondID[length-j]; //the chunk after "+"
            int asterPlace = 0;
            int aster2Place = 0;
            for(int p = 0; p < j; p++) {
              firstID[p] = 0; //initialize each char element
              firstID[p] = photoID[i][p]; //set each to each char before j (before "+")
            }
            for(int p = 0; p < j; p++) {
              if(firstID[p] == 42) {
              //  printf("This reaction has a multiplier: %s\n", photoID[i]);
                asterPlace = p;
                //42 is the ascii code for "*"
                //p is now location of "*"
                //then this photolysis param has a special multiplier that will affect its rate
                //reset paramMults, paramMult[1] is already 0, so just set first
                paramMult[0][i] = 0;
                //find decimal place
                int decimalPlace = -1;
                for(int x = 0; x < p; x++) {
                  if(firstID[x] == 46) {
                    //42 is the ascii code for "."
                    decimalPlace = x;
                  }
                }
                for(int x = 0; x < decimalPlace; x++) {
                  //46 is the ascii code for "."
                  paramMult[0][i] = (firstID[x] - 48)/pow(10, x - decimalPlace + 1);
                }
                for(int x = p - 1; x > decimalPlace; x--) {
                  //coming down from location of "*" to "."
                  paramMult[0][i] = (firstID[x] - 48)/pow(10, x - 1);
//                  printf("p = %d, dec = %d, x = %d, mult = %f\n", p, decimalPlace, x, paramMult[0][i]); 
                } //end "." multipliers
              } // end check if first chunk of "+" param has "*"
              else {
              //this reaction has no special multipliers, compare reaction param requirement
              //and pair with necessary parameter for paramNumID[1][i]

              setParam = paramNumID[0][i];
                if(paramLength == j) {
                  for(int x = 0; x < j ; x++) {
                    if(firstID[x] == pparamID[n][x]) {
                      //appointing this reaction with a photolytic parameter within rateLibrary_atmosCHASER_photo.data
                      paramNumID[0][i] = n;
//                      printf("comparing %s with: %s\n", firstID, pparamID[n]);
                    } else {
                      paramNumID[0][i] = setParam;
                      break;
                    }
                  } //end cycle through each char of first chunk of photoID[i] with "+"
                }
//                printf("Final paramNumID[0]: %d\n", paramNumID[0][i]);
              } //end else: deal with this firstID, first chunk of a photoID[i] with "+" with no special multipliers
            } //end cycle through chars of first element from params with "+"
            //if an asterisk was found, a multiplier was registered, now we must connect parameter to reaction
            if(asterPlace > 0){
              setParam = paramNumID[0][i];
              for(int x = asterPlace+1; x < j ; x++) {
                if(firstID[x] == pparamID[n][x-asterPlace-1]) {
                  paramNumID[0][i] = n;
                } else {
                  paramNumID[0][i] = setParam;
                  break;
                }
              } //end cycle through each char of first chunk of photoID[i] with "+", connecting param to reaction
//                printf("Final paramNumID[0] with multiplier: %d\n", paramNumID[0][i]);
            }//end check if this reaction had a multiplier
            for(int p = 0; p < length-j; p++) {
              secondID[p] = 0;
              secondID[p] = photoID[i][j+p+1];
            }
            for (int p = 0; p < length-j; p++) {
              if(secondID[p] == 42) {
                //42 is the ascii code for "*"
                //now p is the location of the "*"
                aster2Place = p;
                //then this photolysis param has a special multiplier that will affect its rate
                //find decimal place
                int decimalPlace = -1;
                for(int x = 0; x < p; x++) {
                  if(secondID[x] == 46) {
                    decimalPlace = x;
                  }
                }
                for(int x = 0; x < decimalPlace; x++) {
                  //46 is the ascii code for "."
                  paramMult[1][i] = (secondID[x] - 48)/pow(10, x - decimalPlace + 1);
                }
                for(int x = p - 1; x > decimalPlace; x--) {
                  //coming down from location of "*" to "."
                  paramMult[1][i] = (secondID[x] - 48)/pow(10, x - 1);
                }//end "." multipliers
//                printf("p = %d, dec = %d, mult = %f\n", p, decimalPlace, paramMult[1][i]); 
              } // end "*" params
              else {
              //this reaction has no special multipliers, compare reaction param requirement
              //and pair with necessary parameter for paramNumID[1][i]
                if(aster2Place == 0){
                  paramMult[1][i] = 1;
                }
                setParam = paramNumID[1][i];
                if(paramLength == length-j-1) {
                for(int x = 0; x < length-j-1 ; x++) {
                  if(secondID[x] == pparamID[n][x]) {
                    paramNumID[1][i] = n;
                  } else {
                    paramNumID[1][i] = setParam;
                    break;
                  }
                } //end cycle through each char of first chunk of photoID[i] with "+"
                }
//                printf("Final paramNumID[1]: %d\n", paramNumID[1][i]);
              }
            }
            if(aster2Place > 0){
              setParam = paramNumID[1][i];
              for(int x = aster2Place+1; x < length-j ; x++) {
                if(secondID[x] == pparamID[n][x-aster2Place-1]) {
                  paramNumID[1][i] = n;
                } else {
                  paramNumID[1][i] = setParam;
                  break;
                }
              } //end cycle through each char of first chunk of photoID[i] with "+", connecting param to reaction
//                printf("Final paramNumID[1] with multiplier: %d\n", paramNumID[1][i]);
            }//end check if this reaction had a multiplier
          } // end "+" separator. All other reactions only have one param
          else {
            //this reaction does not have a "+", so only need
            //to assign a single parameter(ID) to this reaction
            setParam = paramNumID[0][i];
  //          if(i==12)
//            printf("setParam: %d\n",setParam);
            //find length of current parameter we want to compare to, and choose to cycle over greater length in next for loop for comparison. 
            //if different lengths, not the same, so we can skip. If same length, compare each letter to ensure they are indeed the same.
            if(paramLength == length) {
            for(int x = 0; x < length ; x++) {
              if(photoID[i][x] == pparamID[n][x]) {
                paramNumID[0][i] = n;
  //              if(i==12)
//                printf("comparing requiredreaction ID[%d][%d]: %s to available parameter ID[%d][%d]: %s\n", i, x, photoID[i], n, x, pparamID[n]); 
              } //end compare each char of photoID[i] and pparamID[n] 
              else {
                paramNumID[0][i] = setParam;
                break;
              } //end else: pparamID and photoID have a different char
            } //end cycle through each char of photoID[i] which has length = length
            }
          } //end else: this photoID[i] does not have a "+"
        } //end cycle through chars[j] of photoID[i], the photolytic reaction required parameter ID
      } //end check if this reaction needs a parameter (!= 1)
                //    if(i==12) {
                  //  printf("i:%d\n", i);
                    //  printf("fromNetwork - paramNumID: %d\n", paramNumID[0][i]);
                   // }
    } //end cycle through all photolytic reactions, checking if needs param
  } //end cycle through all photolytic parameters, picking up reactions to be connected

  /*
  for(int i = 0; i < photolytic; i++) {
    printf("PReaction[%d] needs param[0] = %f*%s(%d) and param[1] = %f*%s(%d)\n", i, paramMult[0][i], pparamID[paramNumID[0][i]], paramNumID[0][i], paramMult[1][i], pparamID[paramNumID[1][i]], paramNumID[1][i]);
  }
  */

} //end loadPhotolytic

void Network::parseFlux(int *numProducts, int **reactant, vec_4i *reactantC,
	int **product, vec_4i *productC)
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
//        printf("reactant[%d][%d]: %d, species[%d]\n", k, j, reactant[k][j], i);
				if (i == reactant[k][j])
					totalL++;
			}
			
			// Loop over products for this reaction
			for (int k = 0; k < numProducts[j]; k++)
			{
				if (i == product[k][j])
					totalR++;
			}
			
			total = totalL - totalR;
			
			if (total > 0)       // Contributes to F- for this isotope
			{
				numFminus++;
				reacMask[i + species * j] = -total;
				tempInt2[incrementMinus + numFminus - 1] = j;
				 if (showParsing)
				 	printf("%s reacIndex=%d %s nReac=%d nProd=%d totL=%d totR=%d tot=%d F-\n",
				 		   speciesLabel[i], j, reactionLabel[j], numReactingSpecies[j], numProducts[j], totalL,
				 		   totalR, total);
			}
			else if (total < 0)  // Contributes to F+ for this isotope
			{
				numFplus++;
				reacMask[i + species * j] = -total;
//        printf("reacMask[%d + %d * %d] = %d\n", i, species, j, reacMask[i+species*j]);
				tempInt1[incrementPlus + numFplus - 1] = j;
				 if (showParsing)
				 	printf("%s reacIndex=%d %s nReac=%d nProd=%d totL=%d totR=%d tot=%d F+\n",
				 		   speciesLabel[i], j, reactionLabel[j], numReactingSpecies[j], numProducts[j], totalL,
				 		   totalR, total);
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
		
		if (showParsing == 1)
		 	printf("%d %s numF+ = %d numF- = %d\n", i, speciesLabel[i], numFplus, numFminus);
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
  //check how many species have no Fplus between the last one that did and current one.
  int checkNumSinceNonZeroPlus = 0;
  int checkNumSinceNonZeroMinus = 0;
	for (int i = 1; i < species; i++)
	{
    if(numFluxPlus[i] != 0) {
  		FplusIsotopeCut[i] = numFluxPlus[i] + FplusIsotopeCut[i - checkNumSinceNonZeroPlus - 1];
    } else {
      FplusIsotopeCut[i] = 0;
    }

    if(numFluxMinus[i] != 0) {
  		FminusIsotopeCut[i] = numFluxMinus[i] + FminusIsotopeCut[i - checkNumSinceNonZeroMinus - 1];
    } else {
      FminusIsotopeCut[i] = 0;
    }

    if(numFluxPlus[i] == 0) {
      checkNumSinceNonZeroPlus++;
    } else {
      checkNumSinceNonZeroPlus = 0;
    }

    if(numFluxMinus[i] == 0) {
      checkNumSinceNonZeroMinus++;
    } else {
      checkNumSinceNonZeroMinus = 0;
    }

 //   printf("Starting Position for isotope[%d] in Fplus:%d, Fminus: %d, Cplus:%d Cminus:%d lastPlus:%d lastMinus:%d\n",i, FplusIsotopeCut[i], FminusIsotopeCut[i], checkNumSinceNonZeroPlus, checkNumSinceNonZeroMinus, i - checkNumSinceNonZeroMinus,i - checkNumSinceNonZeroMinus);
	}
	
	int currentIso = 0;
  int lastIsoWFplus = 0;
  int lastIsoWFminus = 0;
  int setj = 0;
	for (int i = 0; i < species; i++)
	{
    //must compensate for isotopes that have no flux, only include those that do. 
    if(FplusIsotopeCut[i] != 0) {
      //then this isotope has some Fplus's, Set those Fplus's to this isotope
      if(i == 0) {
        setj = 0;
      }
      else {
        setj = FplusIsotopeCut[lastIsoWFplus];
      }
      for(int j = setj; j < FplusIsotopeCut[i]; j++) {
        FplusIsotopeIndex[j] = i;
        //printf("This species[%d] ends its Fplus's at Fplus[%d], Setting Fplus[%d] to species[%d], lastIso: %d\n", i, FplusIsotopeCut[i], j, FplusIsotopeIndex[j], lastIsoWFplus);
      }
      lastIsoWFplus = i;
    }
    if(FminusIsotopeCut[i] != 0) {
      //then this isotope has some Fplus's, Set those Fplus's to this isotope
      if(i == 0) {
        setj = 0;
      }
      else {
        setj = FminusIsotopeCut[lastIsoWFminus];
      }
      for(int j = setj; j < FminusIsotopeCut[i]; j++) {
        FminusIsotopeIndex[j] = i;
        //printf("This species[%d] ends its Fminus's at Fminus[%d], Setting Fminus[%d] to species[%d], lastIso: %d\n", i, FminusIsotopeCut[i], j, FminusIsotopeIndex[j], lastIsoWFminus);
      }
      lastIsoWFminus = i;
    }
//  printf("This Fplus %d corresponds to this speices %d\n", i, FplusIsotopeIndex[i]);
	}
	
	for (int i = 0; i < totalFplus; i++)
	{
		MapFplus[i] = tempInt1[i];
//    printf("This Fplus[%d] was caused by this reac[%d]\n", i, MapFplus[i]);
	}
	
	for (int i = 0; i < totalFminus; i++)
	{
		MapFminus[i] = tempInt2[i];
//    printf("This Fminus[%d] was caused by this reac[%d]\n", i, MapFminus[i]);
	}
	
	
  //compensate for possible zero fluxes, if a species has no +/- flux
  lastIsoWFplus = 0;
  lastIsoWFminus = 0;
  if(numFluxPlus[0] !=0) {
  	FplusMin[0] = 0;
	  FplusMax[0] = numFluxPlus[0] - 1;
    //printf("Species[%d]'s Fplus's can be found starting with Fplus[%d] and ending with Fplus[%d], numFluxplus: %d, lastFplus: %d\n", 0, FplusMin[0], FplusMax[0], numFluxPlus[0], lastIsoWFplus);
  } else {
  	FplusMin[0] = 0;
	  FplusMax[0] = 0;
  }
	for (int i = 1; i < species; i++)
	{
    if(numFluxPlus[i] !=0) {
		  FplusMin[i] = FplusMax[lastIsoWFplus] + 1;
  		FplusMax[i] = FplusMin[i] + numFluxPlus[i] - 1 ;
      lastIsoWFplus = i;
    } else {
      //give this species nothing to iterate over
  	  FplusMin[i] = 0;
  	  FplusMax[i] = 0;
    }
      //printf("Species[%d]'s Fplus's can be found starting with Fplus[%d] and ending with Fplus[%d], numFluxPlus: %d, lastFplus: %d\n", i, FplusMin[i], FplusMax[i], numFluxPlus[i], lastIsoWFplus);
	}
	// Populate the FminusMin and FminusMax arrays
  if(numFluxMinus[0] !=0) {
	  FminusMin[0] = 0;
  	FminusMax[0] = numFluxMinus[0] - 1;
    //printf("Species[%d]'s Fminus's can be found starting with Fminus[%d] and ending with Fminus[%d], numFluxminus: %d, lastFminus: %d\n", 0, FminusMin[0], FminusMax[0], numFluxMinus[0], lastIsoWFminus);
  } else {
  	FminusMin[0] = 0;
    FminusMax[0] = 0;
  }
	for (int i = 1; i < species; i++)
	{
    if(numFluxMinus[i] !=0) {
  		FminusMin[i] = FminusMax[lastIsoWFminus] + 1;
	  	FminusMax[i] = FminusMin[i] + numFluxMinus[i] - 1 ;
      lastIsoWFminus = i;
    } else {
  	  FminusMin[i] = 0;
  	  FminusMax[i] = 0;
    }
    //printf("Species[%d]'s Fminus's can be found starting with Fminus[%d] and ending with Fminus[%d], numFluxMinus: %d, lastFminus: %d\n", i, FminusMin[i], FminusMax[i], numFluxMinus[i], lastIsoWFminus);
	}
	
	// Populate the FplusFac and FminusFac arrays that hold the factors counting the
	// number of occurences of the species in the reaction.  Note that this can only
	// be done after parseF() has been run to give reacMask[i][j].
	
	int tempCountPlus = 0;
	int tempCountMinus = 0;
	for (int i = 0; i < species; i++)
	{
    //printf("species: %d\n", i);
		for (int j = 0; j < reactions; j++)
		{
      //generate multiplying factor for fractional coefficients of this species in this reaction
      fern_real fracCoeff = 1.0;
      //multiply coefficients from reactants
      for(int k = 0; k < numReactingSpecies[j]; k++) {
        if(reactant[k][j] == i) {
          fracCoeff *= reactantC[j][k]; 
        }
      }
      for(int k = 0; k < numProducts[j]; k++) {
        if(product[k][j] == i) {
          fracCoeff *= productC[j][k]; 
        }
      }
       //printf("FracCoeff for species[%d] due to reaction[%d] is %f\n", i, j, fracCoeff);

			if (reacMask[i + species * j] > 0)
			{
        //printf("reacmask[%d+%d*%d] = %d\n", i, species, j, reacMask[i + species * j]);
        //multiplying fracCoeff for fractional coefficients of species in certain reactions. Most = 1.0
				FplusFac[tempCountPlus] = (fern_real)reacMask[i + species * j] * fracCoeff;
        //printf("FplusFac[%d] (spec=%d, reac=%d): %f\n", tempCountPlus, i, j, FplusFac[tempCountPlus]);
				tempCountPlus++;
			}
			else if (reacMask[i + species * j] < 0)
			{
        //multiplying fracCoeff for fractional coefficients of species in certain reactions. Most = 1.0
        //printf("reacminusmask[%d+%d*%d] = %d\n", i, species, j, reacMask[i + species * j]);
				FminusFac[tempCountMinus] = -(fern_real)reacMask[i + species * j] * fracCoeff;
        //printf("FMinusFac[%d] (spec=%d, reac=%d): %f\n", tempCountPlus, i, j, FminusFac[tempCountMinus]);
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
	
}


void Network::allocate()
{
	// Allocate the network data
	
	Z = new unsigned char[species];
	N = new unsigned char[species];
	
	FplusMax = new unsigned short [species];
	FminusMax = new unsigned short [species];
	FplusMin = new unsigned short [species];
	FminusMin = new unsigned short [species];
	
	
	// Allocate the reaction data
	
	for (int i = 0; i < 19; i++)
		P[i] = new fern_real[reactions];
	
	numReactingSpecies = new unsigned char[reactions];
	statFac = new fern_real[reactions];
	Q = new fern_real[reactions];
  RGmemberIndex = new int [reactions];
  isReverseR = new int [reactions];
  reacType = new int [reactions];
  PEnumProducts = new int[reactions];
  ReacParent = new int [reactions];
  RGid = new int [numRG];
  ReacGroups = new int[reactions];

  for (int i = 0; i < 10; i++) {
    product[i] = new int[reactions];
		reactant[i] = new int[reactions];
  }

  for (int j = 0; j < 49; j++) {
		aparam[j] = new fern_real[photoparams];
  }

	for (int i = 0; i < 2; i++) {
		paramNumID[i] = new int[photolytic];
		paramMult[i] = new fern_real[photolytic];
  }

  pEquil = new int [numRG];
}

void Network::setSizes(const Network &source)
{
	species = source.species;
	reactions = source.reactions;
	totalFplus = source.totalFplus;
	totalFminus = source.totalFminus;
	photoparams = source.photoparams;
	photolytic = source.photolytic;
	numRG = source.numRG;
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
	
	for (int n = 0; n < 19; n++)
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
	
	for (int n = 0; n < 10; n++)
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
