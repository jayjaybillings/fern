#include <stdio.h>
#include <cmath>
#include <ctime>
#include <time.h>
#include <string>
#include <sstream>
#include <cstring>
#include <iostream>
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
	const unsigned short numberPhotoParams = network.photoparams;
	const unsigned short numberPhotolytic = network.photolytic;

	unsigned char *Z;
	unsigned char *N;

	fern_real *FplusFac;
	fern_real *FminusFac;

	unsigned short *MapFplus;
	unsigned short *MapFminus;

	unsigned short *FplusMax;
	unsigned short *FminusMax;
	unsigned short *FplusMin;
	unsigned short *FminusMin;

	const fern_real massTol = network.massTol;
	const fern_real fluxFrac = network.fluxFrac;

	/* Declare pointer variables for IntegrationData arrays.  */

	fern_real *Y;

	//DSOUTPUT
	const bool plotOutput = 0;
const bool GNUplot = 0;
const bool plotRadicals = 0;
	const int numIntervals = 100;
	int plotStartTime = 0;
	fern_real intervalLogt;
  fern_real nextOutput = 0.0;
  int chooseyourSpecies = 1000; //for GNUplotting of rates and fluxes by species
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
  int *reacType;
  reacType = network.reacType;

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
	FplusMin = network.FplusMin;
	FminusMin = network.FminusMin;

	/* Assign IntegrationData pointers. */
	
	Y = integrationData.Y;

	fern_real maxFlux;
	fern_real sumX = 0.0;
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
  /* Altered for Atmospheric Chemistry */

	/*
	   Compute the temperature-dependent factors for the rates.
	   Since we assume the GPU integration to be done at constant
	   temperature and density, these only need be calculated once
	   per GPU call.
	*/
	bool displayPhotodata = false;

  fern_real T = integrationData.T;
  fern_real H2O = integrationData.H2O;
  fern_real M = integrationData.M;
  fern_real Patm = integrationData.Patm;
  fern_real alt = integrationData.alt;
  fern_real zenith = integrationData.zenith;
  fern_real pmb = integrationData.pmb; //air pressure in millibar
//  parameters for calculating dynamic solar zenith
  fern_real latitude = 35.97; //Knoxville Lat
  fern_real longitude = -83.94; //Knoxville Long
  time_t now = time(0);
  int month = 2;
  int months[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  int daymonth = 4;
  int year = 2016;
  int hour = 19; //12:00am (midnight) 24hr clock
  int minute = 0;
  int second = 0;
  int tzone = -5; //Offset from GMT

  fern_real doy = 0.0;//day-of-year
  //calculate day of year
  for(int i = 0; i < month; i++) {
    doy = doy + months[i];
  }
  doy = doy + daymonth; //add days elapsed in this month
  //end calculate day of year

  //calculate UTC hour in decimal format
  fern_real hrlocal = hour + (minute/60) + (second/3600);
  fern_real hrutc = hrlocal - tzone;

  fern_real delta_year = year-1949; //years since 1949

  fern_real leap = delta_year/4;

  //modified Julian Day to account for leap days
  fern_real mjd = 32916.5+delta_year*365+leap+doy+hrutc/24;

  if (year % 100 == 0 && year % 400 != 0) {
    mjd = mjd-1;
  }

  // time in days since Jan 1, 2000 Greenwich Noon
  fern_real time21 = mjd-51545.0;

  // time in seconds since Jan 1, 2000 Greenwich Noon for simulation time
  fern_real tstart = time21*24*3600;
  fern_real td = t/(24*3600); 

/*  std::ostringstream timedeets;
  timedeets << year << "-" << month << "-" << daymonth << " " << hour << ":" << minute << ":" << second;
  struct tm time_thing;
  char *date_thing = new char[1024];
  strcpy(date_thing, timedeets.str().c_str());
  strptime(date_thing,"%Y-%m-%d %H:%M:%S", &time_thing); 
  time_t epochtime = mktime(&time_thing);
  printf("time: %s\n", timedeets.str().c_str());
  printf("epoch: %d\n", epochtime);

  delete date_thing;
*/

  //convert Y[i] from parts-per-billion to molecules/cm^3
  //necessary for calculation using involved rates
  fern_real cair = pmb*7.2428e+18/T; //concentration of air in molecules/cm^3
  for(int i = 0; i < numberSpecies; i++) {
//    printf("Y[%d](ppb): %e\n", i, Y[i]);
    Y[i] = (Y[i]/1e9)*cair; //Y[i] in molecules/cm^3
//    printf("Y[%d](m/cm^3): %e\n", i, Y[i]);
  } 
  

	for (int i = 0; i < network.reactions; i++) {
    if(reacType[i] == 2) {
      calculatePhotolyticRates(i, zenith, alt, network.paramNumID, network.aparam, network.paramMult, Rate);



      /***** FOR GRAPHING CONSTANT PHOTOLYTIC REACTION RATES VS TEMP (independent of temp... this is just for reference) IN GNUPLOT *****/
/*
      //to graph reactions that contain any but only of these species
      //int isInReac = 1;
      //to graph reactions that contain any of these species
      int isReactant = 0;
      int isProduct = 0;
      for(int j = 0; j < network.numReactingSpecies[i]; j++) {
        int m = network.reactant[j][i];
        //printf("Reaction[%d], with numreacspec[%d] uses species[%d]\n", i, network.numReactingSpecies[i], network.reactant[j][i]);
        //if((m == 0 || m == 2 || m == 3 || m == 6 || m == 7 || (m >= 12 && m <= 16) || m == 18 || m == 19 || m == 22 || m == 27 || (m >= 75 && m <= 77) || m == 80 || m == 85 || (m >= 95 && m <= 97) || m == 99 || m == 101) && (B > 0 || C > 0 || D > 0 || E > 0 || F > 0)) {
        //plot single species' reaction rates
        if(m == chooseyourSpecies) {
          isReactant = 1;
        } else {
          //to graph reactions that contain any but only of these species
          //isReactant = 0;
        }
      }

      //Also, if this reaction generates this species, also plot that, but with a different color.
      for(int j = 0; j < network.PEnumProducts[i]; j++) {
        int m = network.product[j][i];
        //printf("Reaction[%d], with numProducts[%d] generates species[%d]\n", i, network.PEnumProducts[i], network.product[j][i]);
        //if((m == 0 || m == 2 || m == 3 || m == 6 || m == 7 || (m >= 12 && m <= 16) || m == 18 || m == 19 || m == 22 || m == 27 || (m >= 75 && m <= 77) || m == 80 || m == 85 || (m >= 95 && m <= 97) || m == 99 || m == 101) && (B > 0 || C > 0 || D > 0 || E > 0 || F > 0)) {
        //plot single species' reaction rates
        if(m == chooseyourSpecies) {
          isProduct = 1;
        } else {
          //to graph reactions that contain any but only of these species
          //isProduct = 0;
        }
      }

      //GNUPLOT RATES V TEMP
      if(isReactant == 1 || isProduct == 1) {
        printf("%e ", Rate[i]);
      } 
        if(isReactant == 1 && isProduct == 0) {
          printf(" title 'J[%d]' w linespoints lt rgb 'green', \\\n",i);
        } else if (isReactant == 0 && isProduct == 1) {
          printf(" title 'J[%d]' w linespoints lt rgb 'purple', \\\n",i);
        }
*/
    } else if(reacType[i] == 0) {
      //then this is a regular chemical reaction (not M-type or Photolytic)
      fern_real A = network.P[0][i];
      fern_real B = network.P[1][i];
      fern_real C = network.P[2][i];
      fern_real D = network.P[3][i];
      fern_real E = network.P[4][i];
      fern_real F = network.P[5][i];
      fern_real G = network.P[6][i];
      fern_real a = network.P[7][i];
      fern_real b = network.P[8][i];
      fern_real c = network.P[9][i];
      fern_real d = network.P[10][i];
      fern_real e = network.P[11][i];
      fern_real t = network.P[12][i];
      fern_real u = network.P[13][i];
      fern_real v = network.P[14][i];
      fern_real w = network.P[15][i];
      fern_real x = network.P[16][i];
      fern_real Q = network.P[17][i];
      fern_real R = network.P[18][i];
		  #ifdef FERN_SINGLE
      fern_real p1 = 1 + B*H2O*exp(a/T);
      fern_real p2 = C*exp(b/T)*(v+w*M);
      fern_real p3 = E*exp(d/T);
      fern_real p4 = pow(D*exp(c/T), x);
      fern_real p5 = F*pow(T,2)*exp(e/T);
		  #else
      fern_real p1 = 1 + B*H2O*expf(a/T);
      fern_real p2 = C*expf(b/T)*(v+w*M);
      fern_real p3 = E*expf(d/T);
      fern_real p4 = powf(D*expf(c/T), x);
      fern_real p5 = F*powf(T,2)*expf(e/T);
		  #endif
      fern_real p6 = G*(1+R*Patm);
      //Specific parameterization for special formats.
      if(i == 36) {
//        printf("A: %e, B: %e, C: %e, D: %e, E: %e, F: %e, G: %e, a: %e, b: %e, c: %e, d: %e, e: %e, t: %e, u: %e, v: %e, w: %e, x: %e, Q: %e, R: %e\n", A, B, C, D, E, F, G, a, b, c, d, e, t, u, v, w, x, Q, R);
        //This reaction has the form k11 = (ka + kb[M])*kc
        Rate[i] = (p3+p2)*p1;       
      } else if(i == 38) {
        //This reaction has the form k13 = ka + kb [M]/(1 + kb [M]/kc)
		    Rate[i] = p3+(p2/(1+p2*p4));
      } else {
  		  Rate[i] = A + ((Q * p1 * (p2 + p3)) / (1 + (t + u * p2) * p4)) + p5 + p6;
      }
      /*
      printf("%e + ((%f * (1 + %e * H2O * exp(%f/T)) * (%eexp(%f/T) * (%f + %f*M)+%eexp(%f/T)))/(%f + (%f * %eexp(%f/T) * (%f + %f*M)) * pow(%eexp(%f/T), %f))) + %e*pow(T,2)*exp(%f/T) + %e*(1+%fPatm)\n", 
        A, Q, B, a, C, b, v, w, E, d, t, u, C, b, v, w, D, c, x, F, e, G, R);
      */

      /***** FOR GRAPHING REACTION RATES VS TEMP IN GNUPLOT *****/
      //to graph reactions that contain any but only of these species
      //int isInReac = 1;
      //to graph reactions that contain any of these species
      int isReactant = 0;
      int isProduct = 0;
      for(int j = 0; j < network.numReactingSpecies[i]; j++) {
        int m = network.reactant[j][i];
        //printf("Reaction[%d], with numreacspec[%d] uses species[%d]\n", i, network.numReactingSpecies[i], network.reactant[j][i]);
        //if((m == 0 || m == 2 || m == 3 || m == 6 || m == 7 || (m >= 12 && m <= 16) || m == 18 || m == 19 || m == 22 || m == 27 || (m >= 75 && m <= 77) || m == 80 || m == 85 || (m >= 95 && m <= 97) || m == 99 || m == 101) && (B > 0 || C > 0 || D > 0 || E > 0 || F > 0)) {
        //plot single species' reaction rates
        if((m == chooseyourSpecies) && (A > 0 || B > 0 || C > 0 || D > 0 || E > 0 || F > 0)) {
          isReactant = 1;
        } else {
          //to graph reactions that contain any but only of these species
          //isReactant = 0;
        }
      }

      //Also, if this reaction generates this species, also plot that, but with a different color.
      for(int j = 0; j < network.PEnumProducts[i]; j++) {
        int m = network.product[j][i];
        //printf("Reaction[%d], with numProducts[%d] generates species[%d]\n", i, network.PEnumProducts[i], network.product[j][i]);
        //if((m == 0 || m == 2 || m == 3 || m == 6 || m == 7 || (m >= 12 && m <= 16) || m == 18 || m == 19 || m == 22 || m == 27 || (m >= 75 && m <= 77) || m == 80 || m == 85 || (m >= 95 && m <= 97) || m == 99 || m == 101) && (B > 0 || C > 0 || D > 0 || E > 0 || F > 0)) {
        //plot single species' reaction rates
        if((m == chooseyourSpecies) && (A > 0 || B > 0 || C > 0 || D > 0 || E > 0 || F > 0 || G > 0)) {
          isProduct = 1;
        } else {
          //to graph reactions that contain any but only of these species
          //isProduct = 0;
        }
      }

      //GNUPLOT RATES V TEMP
      if(isReactant == 1 || isProduct == 1) {
        if(B>0) {
          printf("(1+%e*%e*exp(%e/x))*(%e*%e*exp(%e/x)+%e*exp(%e/x))", B, H2O, a, C, M, b, E, d);
        } else {
          printf("%e+((%e*(%e*exp(%e/x)*(%e+%e*%e)+%e*exp(%e/x)))/(%e+(%e*%e*exp(%e/x)*(%e+%e*%e)*(%e*exp(%e/x))**%e)))+(%e*(x**2)*exp(%e/x))+(%e*(1+%e*%e))",A,Q,C,b,v,w,M,E,d,t,u,C,b,v,w,M,D,c,x,F,e,G,R,Patm);
        } 
        if(isReactant == 1 && isProduct == 0) {
          printf(" title 'K[%d]' w linespoints lt rgb 'red', \\\n",i-25);
        } else if (isReactant == 0 && isProduct == 1) {
          printf(" title 'K[%d]' w linespoints lt rgb 'blue', \\\n",i-25);
        }
      }

      /***** END FOR GRAPHING REACTION RATES VS TEMP IN GNUPLOT *****/


      if(displayPhotodata)
        printf("Chemical Rate[%d] = %e\n", i-25, Rate[i]);
    } else if(reacType[i] == 1) {
      //then this is an Mtype reaction
      fern_real A = network.P[0][i];
      fern_real B = network.P[1][i];
      fern_real C = network.P[2][i];
      fern_real a = network.P[3][i];
      fern_real b = network.P[4][i];
      fern_real x = network.P[5][i];
      fern_real y = network.P[6][i];
      //check if this reaction is a reverse M-type reaction of previous reaction
      //Depends on Mtype RG members to be in sequence, 
      //forward then reverse as deemed by CHASER paper
      if(ReacParent[i] == ReacParent[i-1]) {
		    #ifdef FERN_SINGLE
          fern_real p1 = A*exp(B/T);
        #else
          fern_real p1 = A*expf(B/T);
        #endif

        Rate[i] = Rate[i-1]/p1;
        if(displayPhotodata)
          printf("Reverse MType[%d] = %e\n", i-25, Rate[i]);
      //Mtype Reaction
      } else {
		    #ifdef FERN_SINGLE
          fern_real p1 = A*pow((a/T),x);
          fern_real p2 = B*pow((b/T),y);
          fern_real p3 = pow((log10(p1*M/p2)),2);
          fern_real p4 = 1/(1+p3);
          fern_real p5 = pow(C, p4);
        #else
          fern_real p1 = A*powf((a/T),x);
          fern_real p2 = B*powf((b/T),y);
          fern_real p3 = powf((log10(p1*M/p2)),2);
          fern_real p4 = 1/(1+p3);
          fern_real p5 = powf(C, p4);
        #endif
        fern_real rk = p1*M/p2;
        fern_real logrk = log10(rk);
        fern_real g = 1/(1+(logrk*logrk));
//        printf("Reac[%d], k0:%e, ki:%e, log^2:%e, 1/1+log:%e, F^1/1+log:%e\n", i, p1, p2, p3, p4, p5);
//        Rate[i] = (p1*M/(1+(p1*M/p2)))*p5;
        Rate[i] = 0;
        Rate[i] = (p1*M/(1+rk))*pow(C,g);
        if(displayPhotodata)
          printf("Parent/Solo MType[%d] = %e\n", i-25, Rate[i]);
      }
    }
	}

	/* Author: Daniel Shyles */
	/* Begin Partial Equilibrium calculation */
	const bool displayRGdata = false;
    fern_real kf;
    fern_real kr;
    fern_real *final_k[2];
	int countRG = 0;
		//first set up array of final reaction rates for each RG based on Rate[i] calculated above
		for(int m = 0; m < 2; m++)
			final_k[m] = new fern_real [network.numRG];
		if(displayRGdata)
			printf("Start Reaction Group Data\nNumber Reaction Groups: %d\n\n",network.numRG);

		//second to calculate final reaciton rates for each RG
		for(int i = 0; i < network.reactions; i++) {
			if(displayRGdata && ReacGroups[i] != 0) {
				printf("RG #%d\nRG Class: %d\nParent Reaction: %d\n", countRG, ReacGroups[i], i);
				//output numReactingSpecies and numProducts for Parent of RG
                printf("numReacting: %d, numProducts: %d\n", network.numReactingSpecies[i], network.PEnumProducts[i]);			
				if(ReacGroups[i] == 1) {		
					printf("Reactant SID: %d; Product SID: %d\n",network.reactant[0][i], network.product[0][i]);
				} else if(ReacGroups[i] == 2) {
            printf("Reactant SID: %d, %d; Product SID: %d\n",network.reactant[0][i], network.reactant[1][i], network.product[0][i]);
        } else if(ReacGroups[i] == 3) {
            printf("Reactant SID: %d, %d, %d; Product SID: %d\n",network.reactant[0][i], network.reactant[1][i], network.reactant[2][i], network.product[0][i]);
        } else if(ReacGroups[i] == 4) {
            printf("Reactant SID: %d, %d; Product SID: %d, %d\n",network.reactant[0][i], network.reactant[1][i], network.product[0][i], network.product[1][i]);
        } else if(ReacGroups[i] == 5) {
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


    //Set initial abundances for O2, N2 and H2:

    
    Y[2] = 0.21*cair; //O2
    Y[97] = 500.0e-9*cair; //H2
    Y[98] = 0.78*cair; //N2
    Y[95] = 3.87e17;


	/*
	   Begin the time integration from t=0 to tmax. Rather than t=0 we
	   start at some very small value of t. This is required for the CUDA C
	   code as well as the Java version.
	*/
	
  t = 1.0e-20;
	dt = integrationData.dt_init;
	timesteps = 1;
	
	fern_real floorFac = 0.1;//does not allow timestep to grow larger than 10% of current time.
	fern_real upbumper = 0.9 * massTol;
	fern_real downbumper = 0.1;
	fern_real massTolUp = 0.25 * massTol;
	fern_real deltaTimeRestart = dt;
	fern_real dtFloor;
	fern_real dtFlux;
	fern_real massChecker;
	
	/* Compute mass numbers and initial mass fractions X for all isotopes. */
	fern_real totalY = 0.0;
	for (int i = 0; i < numberSpecies; i++)
	{
    //printf("Y[%d] = %e\n", i, Y[i]);
    //TODO: Check if this is correct. I'm replacing X[i] by Y[i]/totalY
		//massNum[i] = (fern_real) Z[i] + (fern_real) N[i];
		/* Compute mass fraction X from abundance Y. */
    totalY += Y[i];
  }
	for (int i = 0; i < numberSpecies; i++)
	{ //continued from above. CHECK WITH DR GUIDRY
		//X[i] = massNum[i] * Y[i];
    X[i] = Y[i]/totalY;
	}
	
	sumXLast = NDreduceSum(X, numberSpecies);
	/* Main time integration loop */
	if(plotOutput == 1) {
		printf("STARTOUTPUT\n");//StartOutput
  }
	//holder for Y conversion back to ppb for plotting
  fern_real Yppb = 0.0;
	while (t < integrationData.t_max)
	{
  //calculate zenith angle for new timestamp 
  td = (t+tstart)/(24*3600); 
  zenith = SolarZenithAngle(td, latitude, longitude);
  printf("zenith: %f, dt: %e\n", zenith, dt);

  //recalculate photolytic rates for new zenith angle
	for (int i = 0; i < network.reactions; i++) {
    if(reacType[i] == 2) {
      calculatePhotolyticRates(i, zenith, alt, network.paramNumID, network.aparam, network.paramMult, Rate);
    }
  }    


      //START PLOT INFO//
      //TODO: Update to latest partial equilibrium code from fernPartialEq code.

		if(plotOutput == 1 && log10(t) >= plotStartTime) {
			//Do this once after log10(t) >= plotStartTime.
			if(setNextOut == 0) {
	      intervalLogt = (log10(integrationData.t_max)-log10(t))/numIntervals;
				nextOutput = log10(t);
				setNextOut = 1;
			}
		//stdout to file > fernOut.txt for plottable output
			if(log10(t) >= nextOutput) {
        
        //For GNUplot Output of rates and fluxes
        //if(chooseyourSpecies < 1000)
/*          printf("outputTime: %e \n", t);
          for(int i = 0; i < numberReactions; i++) {
            printf("Rate for Reaction[%d]: %e, Flux: %e\n", i, Rate[i], Flux[i]);
          }
*/
    //REPEAT Flux loops to print for GNUPLOT at proper intervals
		int minny;
    int lastIsoWFplus = 0;
    int lastIsoWFminus = 0;
		for (int i = 0; i < numberSpecies; i++)
		{
      if(i > 0 && FplusMax[i] == 0){
        //printf("Sorry, species[%d] has no F+'s, therefore its FplusSum is: %e\n", i, FplusSum[i]);
      } else {
        minny = (i > 0) ? FplusMax[lastIsoWFplus] + 1 : 0;
			  for (int j = minny; j <= FplusMax[i]; j++)
		  	{
          //printf("Species[%d] will include Fplus[%d]: %e, due to reaaction[%d]\n", i, j, Fplus[j], MapFplus[j]);
          //FOR GNUPLOT OF FLUXES
          if(i == chooseyourSpecies) { //select the species you wish to print fluxes for.. 96=CO2
            //printf("R:%d,  %e ",MapFplus[j],Fplus[j]);
            printf("%e ", Fplus[j]);
          }
  			}
        
        //printf("species[%d]'s Fplus's start at %d and ends at %d, This species has an FplusSum: %e\n", i, minny, FplusMax[i], FplusSum[i]);
        lastIsoWFplus = i;
      }

			/* Serially sum section of F-. */
      if(i > 0 && FminusMax[i] == 0){
        //printf("Sorry, species[%d] has no F-'s, thus, its FminusSum is: %e\n", i, FminusSum[i]);
      } else {
       	minny = (i > 0) ? FminusMax[lastIsoWFminus] + 1 : 0;
  			for (int j = minny; j <= FminusMax[i]; j++)
	  		{
          //printf("Species[%d] will include Fminus[%d]: %e\n", i, j, Fminus[j]);
          if(i == chooseyourSpecies) { //select the species you wish to print fluxes for.. 96=CO2
            //printf("R:%d, %e ",MapFminus[j],Fminus[j]);
            printf("%e ", Fminus[j]);
          }
			  }
        //printf("species[%d]'s Fminus's start at %d and ends at %d, This species has an FminusSum: %e\n", i, minny, FminusMax[i], FminusSum[i]);
        lastIsoWFminus = i;
      }
      //FOR GNUPLOT of fluxes
		}
    //if(chooseyourSpecies < 1000)
     // printf("\n");
      //END GNUPLOT for FLUXES



        if(plotOutput == 1 && GNUplot == 0) {
  		  	printf("OC\n");//OutputCount
        }
				//renormalize nextOutput by compensating for overshooting last expected output time
				nextOutput = intervalLogt+nextOutput;
				asyCount = 0;
				peCount = 0;
        if(plotOutput == 1 && GNUplot == 1) {
          printf("%e ",t);	
        }
				for(int m = 0; m < network.species; m++) {
          //convert back to parts-per-billion for graphing/output
          Yppb = (Y[m]*1e9)/cair; //Y[i] in ppb/cm^3

          /***** FOR GNUPLOT *****/
          if(plotRadicals == 0 && GNUplot == 1) {
            if(m == 0 || m == 12 || m == 6 || m == 13 || m==14 || m==18 || m==27 || m==101) {
              printf("%e ", Yppb);
            }
          } else if (plotRadicals == 1 && GNUplot == 1) {
            //plot shortlived radicals in molecules/cm^3
            if(m == 75 || m == 3 || m == 76 || m == 77) {
              printf("%e ", Y[m]);
            }
          }
          /***** END FOR GNUPLOT *****/

					if(checkAsy(FminusSum[m], Y[m], dt))
						asyCount++;	
				}

          /***** FOR GNUPLOT *****/
          if(plotOutput == 1 && GNUplot == 1) {
            printf("\n");
          }
          /***** END FOR GNUPLOT *****/


        //check frac RG PartialEq
	      partialEquil(Y, numberReactions, ReacGroups, network.reactant, network.product, final_k, pEquil, RGid, numRG, 0.01, eq);
  
				for(int i = 0; i < numRG; i++) {
					if(pEquil[i] == 1)
						peCount++;			
				}
				FracAsy = asyCount/numberSpecies;
				FracRGPE = peCount/numRG;
        if(plotOutput == 1 && GNUplot == 0) {
  				printf("SUD\nti:%edt:%eT9:%erh:%esX:%efasy:%ffrpe:%f\n", t, dt, integrationData.T, integrationData.rho, sumX, FracAsy, FracRGPE);//StartUniversalData
        }
				outputCount++;
			}
		}




		/* Set Yzero[] to the values of Y[] updated in previous timestep. */
		
		for (int i = 0; i < numberSpecies; i++)
		{
			Yzero[i] = Y[i];
      //printf("Y0[%d]: %e\n", i, Yzero[i]);
      //return;
		}
		
		
		/* Compute the fluxes from the previously-computed rates and the current abundances. */
		
		/* Parallel version of flux calculation */
		
		for (int i = 0; i < numberReactions; i++)
		{
      if(reacType[i] != 2) {
	  		int nr = network.numReactingSpecies[i];
  			Flux[i] = Rate[i] * Y[network.reactant[0][i]];
        //printf("reaction[%d] with rate %e causes a flux of %e because we some abundance of species[%d]: %e(first-body)\n", i, Rate[i], Flux[i], network.reactant[0][i], Y[network.reactant[0][i]]);
		  	switch (nr)
	  		{
  			case 3:
				  /* 3-body; flux = rate x Y x Y x Y */
			  	Flux[i] *= Y[network.reactant[2][i]];
		  		Flux[i] *= Y[network.reactant[1][i]];
          //printf("reaction[%d] causes a flux of %e because we some abundance of species[%d]: %e(second) and species[%d]: %e(third-body)\n", i, Flux[i], network.reactant[1][i], Y[network.reactant[2][i]], network.reactant[2][i], Y[network.reactant[2][i]]);
				
  			case 2:
				  /* 2-body; flux = rate x Y x Y */
			  	Flux[i] *= Y[network.reactant[1][i]];
          //printf("reaction[%d] causes a flux of %e because we some abundance of species[%d]: %e(second-body)\n", i, Flux[i], network.reactant[1][i], Y[network.reactant[1][i]]);
	  			break;
  			}
      } else {
        //this is a photolytic reaction, and is first order, ie flux is determined only by the first reactant
        Flux[i] = Rate[i] * Y[network.reactant[0][i]];
        //printf("reaction[%d] with rate %e causes a flux of %e because we some abundance of species[%d]: %e(first-body)\n", i, Rate[i], Flux[i], network.reactant[0][i], Y[network.reactant[0][i]]);
/*        if(i == 23) {
          printf("%d HACET[%e]\n",network.reactant[0][i], Y[network.reactant[0][i]]); 
        }*/
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
    int lastIsoWFplus = 0;
    int lastIsoWFminus = 0;
		for (int i = 0; i < numberSpecies; i++)
		{
      if(i > 0 && FplusMax[i] == 0){
        FplusSum[i] = 0.0;
        //printf("Sorry, species[%d] has no F+'s, therefore its FplusSum is: %e\n", i, FplusSum[i]);
      } else {
        minny = (i > 0) ? FplusMax[lastIsoWFplus] + 1 : 0;
	  		/* Serially sum secction of F+. */
  			FplusSum[i] = 0.0;
			  for (int j = minny; j <= FplusMax[i]; j++)
		  	{
	  			FplusSum[i] += Fplus[j];
          //printf("Species[%d] will include Fplus[%d]: %e, due to reaaction[%d]\n", i, j, Fplus[j], MapFplus[j]);
  			}
        
        //printf("species[%d]'s Fplus's start at %d and ends at %d, This species has an FplusSum: %e\n", i, minny, FplusMax[i], FplusSum[i]);
        lastIsoWFplus = i;
      }

			/* Serially sum section of F-. */
      if(i > 0 && FminusMax[i] == 0){
        FminusSum[i] = 0.0;
        //printf("Sorry, species[%d] has no F-'s, thus, its FminusSum is: %e\n", i, FminusSum[i]);
      } else {
       	minny = (i > 0) ? FminusMax[lastIsoWFminus] + 1 : 0;
			  FminusSum[i] = 0.0;
  			for (int j = minny; j <= FminusMax[i]; j++)
	  		{
		  		FminusSum[i] += Fminus[j];
          //printf("Species[%d] will include Fminus[%d]: %e\n", i, j, Fminus[j]);
			  }
        //printf("species[%d]'s Fminus's start at %d and ends at %d, This species has an FminusSum: %e\n", i, minny, FminusMax[i], FminusSum[i]);
        lastIsoWFminus = i;
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
        if(i==5) {
          //printf("F+: %e, F-: %e\n", FplusSum[i], FminusSum[i]);
        }
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
    //maxFlux is the largest flux in all of the species in the network. Fdiff
    //supposed to disallow more than a fractional change in the abundance of a species in the network.
		dtFloor = floorFac * t; //prevents dt from being larger than 10% of current time
		if (dtFlux > dtFloor) dtFlux = dtFloor; //seeabove^^
			
		dt = dtFlux;

		if (deltaTimeRestart < dtFlux) dt = deltaTimeRestart; //
		
		updatePopulations(FplusSum, FminusSum, Y, Yzero, numberSpecies, dt);
		
		/* Compute sum of mass fractions sumX for all species. */
    //TODO: AGAIN, check with Dr. Guidry if this is the best approach for
    //considering mass fraction with this new definition of Y[i]. Here, Y ≠ X/A
    //as it did in nucleosynthesis case. See ~line 513
		
    totalY = 0;
		for (int i = 0; i < numberSpecies; i++)
		{
      //Recalculate sum of abundances for calculating mass fraction.
      totalY += Y[i];
    }
		for (int i = 0; i < numberSpecies; i++)
		{
			/* Compute mass fraction X from abundance Y. */
      //REMOVED UNTIL I TALK WITH DR. GUIDRY ABOUT NEW WAY OF CALCULATING X
		  //X[i] = massNum[i] * Y[i];
		  X[i] = (Y[i]/totalY);
   // printf("totalY: %f,Y[i]: %e, X[%d}: %e\n", totalY, Y[i],i, X[i]);
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
			massChecker = fabsf(sumXLast - sumX); //TODO due to my sumX always = 1, this = 0. 


			if (test2 > test1 && massChecker > massTol) //false due to sumX=1
			{
				dt *= fmaxf(massTol / fmaxf(massChecker, (fern_real) 1.0e-16), downbumper);
			}
			else if (massChecker < massTolUp) //true due to sumX=1
			{
				dt *= (massTol / (fmaxf(massChecker, upbumper)));
			}
		#else
			fern_real test1 = fabs(sumXLast - 1.0);
			fern_real test2 = fabs(sumX - 1.0);
			massChecker = fabs(sumXLast - sumX);
      //printf("sumxlast: %f, sumx: %f, masschecker: %f\n",sumXLast, sumX, massChecker);

			if (test2 > test1 && massChecker > massTol)
			{
				dt *= fmax(massTol / fmax(massChecker, (fern_real) 1.0e-16), downbumper);
			}
      
			else if (massChecker < massTolUp)
			{
				dt *= (massTol / (fmax(massChecker, upbumper)));
			}
		#endif

      if(dtFlux > dtFloor) {		
	      dt = dtFloor;	
      } else {
        dt = dtFlux;
      }
      
//      printf("maxflux: %e, dt: %e\n", maxFlux, dt);


/*    if(log10(t) < 0) { 
      dt = 0.00001*t;  
    }
    if(log10(t) >= 0 ) { 
      dt = 0.000001*t;  
    }
*/
		updatePopulations(FplusSum, FminusSum, Y, Yzero, numberSpecies, dt);
		
		
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
  totalY = 0;		
	for (int i = 0; i < numberSpecies; i++)
	{
    //TODO: Check if this is correct. I'm replacing X[i] by Y[i]/totalY
		//massNum[i] = (fern_real) Z[i] + (fern_real) N[i];
		/* Compute mass fraction X from abundance Y. */
    totalY += Y[i];
  }
	for (int i = 0; i < numberSpecies; i++)
	{ //continued from above. CHECK WITH DR GUIDRY
		//X[i] = massNum[i] * Y[i];
    X[i] = Y[i]/totalY;
//    printf("totalY: %f,Y[i]: %f, X[%d}: %e\n", totalY, Y[i],i, X[i]);
	}
		
		sumX = NDreduceSum(X, numberSpecies);
		
		
		/* Increment the integration time and set the new timestep. */
			
		t += dt;
		timesteps++;
		
		sumXLast = sumX;
   // printf("sumxlast: %f, sumX: %f\n",sumXLast, sumX);
  /*for(int i = 0; i < numberSpecies; i++) {
    printf("updated Y[%d] for t: %e = %e\n", i, t, Y[i]);
  }*/
	}
  if(plotOutput == 1)
    printf("ENDOUTPUT\n");//EndOutput
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

inline fern_real asymptoticUpdate(fern_real FplusSum, fern_real FminusSum, fern_real Y, fern_real dt)
{
	/* Sophia He formula */
	return (Y + FplusSum * dt) / (1.0 + FminusSum * dt / Y);
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
          //printf("MAXFLUX from species[%d]: %e\n", i, a[i]);
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
    //includes fracCoeff for each species within the reaction
		Fsign[i] = FsignFac[i] * Flux[MapFsign[i]];
  //  printf("Fsign[%d] due to Reaction[%d] with Fac[%e]: %e\n", i, MapFsign[i], FsignFac[i], Fsign[i]);
	}
}

fern_real SolarZenithAngle(fern_real td, fern_real latitude, fern_real longitude) {

  fern_real pi = 2.0*acos(0.0);
  fern_real twopi = 2.0*pi;
  fern_real deg2rad = pi/180.0;
  fern_real mnlong;
  fern_real mnanom;
  fern_real eclong;
  fern_real oblqec;
  fern_real num;
  fern_real den;
  fern_real ra;
  fern_real gmst;
  fern_real lmst;
  fern_real latrad;
  fern_real elc;
  fern_real refrac;
  fern_real soldia;
  fern_real el;
  fern_real dec;
  fern_real ha;  
  fern_real zen;

  // force mean longitude between 0 and 360 degs
  mnlong = 280.460+.9856474*td;
  mnlong = fmod(mnlong,360);
  if (mnlong < 0) {
    mnlong = mnlong+360;
  }

  // mean anomaly in radians between 0 and 2*pi
  mnanom = 357.528+.9856003*td;
  mnanom = fmod(mnanom,360);
  if(mnanom < 0) {
    mnanom=mnanom+360; 
  }
  mnanom=mnanom*deg2rad;

  // compute the ecliptic longitude and obliquity of ecliptic in radians
  eclong = mnlong+1.915*sin(mnanom)+.020*sin(2*mnanom);
  eclong = fmod(eclong,360);
  if (eclong < 0) {
    eclong = eclong+360;
  }
  oblqec = 23.439-.0000004*td;
  eclong = eclong*deg2rad;
  oblqec = oblqec*deg2rad;

  // calculate right ascension and declination
  num = cos(oblqec)*sin(eclong);
  den = cos(eclong);
  ra = atan(num/den);

  // force ra between 0 and 2*pi
  if (den < 0) {
    ra = ra + pi;
  } else if (num < 0) {
    ra = ra + twopi;
  }

  // dec in radians
  dec = asin(sin(oblqec)*sin(eclong));

  // calculate Greenwich mean sidereal time in hours
  // Substitute the approximate gmst formula from the U. S. Naval Observatory web site
  // http://www.usno.navy.mil/USNO/astronomical-applications/astronomical-information-center/approx-sider-time
  gmst = 18.697374558 + 24.06570982441908 * td;

  gmst = fmod(gmst,24);
  if (gmst < 0) {
    gmst = gmst + 24;
  }

  // calculate local mean sidereal time in radians
  lmst = gmst+longitude/15;
  lmst = fmod(lmst,24);
  if (lmst < 0) {
    lmst = lmst + 24;
  }
  lmst = lmst*15*deg2rad;

  // calculate hour angle in radians between -pi and pi
  ha = lmst-ra;
  if (ha < -pi) {
    ha = ha + twopi;
  }
  if (ha > pi) {
    ha = ha - twopi;
  }

  // change latitude to radians
  latrad = latitude*deg2rad;

  // calculate elevation

  el = asin(sin(dec)*sin(latrad)+cos(dec)*cos(latrad)*cos(ha));

  // claculate refraction correction for US stan. atmosphere
  // need to have el in degs before calculating correction

  el = el/deg2rad;

  // note that 3.51823 = 1013.25mb/288K
  if(el >=19.225) {
    refrac = .00452*3.51823/tan(el*deg2rad);
  } else if (el > -.766 && el < 19.225) {
    refrac = 3.51823*(.1594+.0196*el+.00002*pow(el, 2))/(1+.505*el+.0845*pow(el, 2)); 
  } else if (el <= -.766) {
    refrac = 0.0;
  }

  el = el+refrac;

  zen = 90.0 - el;

  return zen;
}

void calculatePhotolyticRates(int i, fern_real zenith, fern_real alt, int **paramNumID, fern_real **aparam, fern_real **paramMult, fern_real *Rate) {
  fern_real cz = cos(zenith);//cosine of zenith angle
  //calculate powers of cz
  fern_real cz2 = cz*cz;
  fern_real cz3 = cz2*cz;
  fern_real cz4 = cz3*cz;
  fern_real cz5 = cz4*cz;
  fern_real cz6 = cz5*cz;
  fern_real zalpha = 1.0;
  int amin = 0;
  int amax = 0;

  if(alt >= 12000) {
    zalpha = 1.0;
    amax = 7; 
    amin = 7; 
  } else if (alt < 12000 && alt >= 10000) {
    zalpha = (12000 - alt)/2000;
    amax = 7; 
    amin = 6; 
  } else if (alt < 10000 && alt >= 8000) {
    zalpha = (10000 - alt)/2000;
    amax = 6; 
    amin = 5; 
  } else if (alt < 8000 && alt >= 6000) {
    zalpha = (8000 - alt)/2000;
    amax = 5; 
    amin = 4; 
  } else if (alt < 6000 && alt >= 4000) {
    zalpha = (6000 - alt)/2000;
    amax = 4; 
    amin = 3; 
  } else if (alt < 4000 && alt >= 2000) {
    zalpha = (4000 - alt)/2000;
    amax = 3; 
    amin = 2; 
  } else if (alt < 2000) {
    zalpha = (2000 - alt)/2000;
    amax = 2; 
    amin = 1; 
  }

  //calculate interpolation parameters
  fern_real zfac = zalpha;
  fern_real zfac1 = 1.0 - zalpha;

  fern_real zpolyhigh = 0;
  fern_real zpolylow = 0;
  fern_real rateparam1 = 0;
  fern_real rateparam2 = 0;
  fern_real amin0 = 0;
  fern_real amin1 = 0;
  fern_real amin2 = 0;
  fern_real amin3 = 0;
  fern_real amin4 = 0;
  fern_real amin5 = 0;
  fern_real amin6 = 0;
  fern_real amax0 = 0;
  fern_real amax1 = 0;
  fern_real amax2 = 0;
  fern_real amax3 = 0;
  fern_real amax4 = 0;
  fern_real amax5 = 0;
  fern_real amax6 = 0;
      /*******************************************************************************
        From Rick's Fortran Code
        CalcJPhoto -  calculates photolysis frequencies (1/s) for a given
        altitude and cosine(zenith angle).
        Photolysis frequencies were generated from TUV 5.0
        (Madronich, S. and S. Flocke, Theoretical estimation of 
        biologically effective UV radiation at the Earth's surface, 
        in Solar Ultraviolet Radiation - Modeling, Measurements and 
        Effects (Zerefos, C., ed.). NATO ASI Series Vol. I52, 
        Springer-Verlag, Berlin, 1997.) as a function of altitude 
        above sea level and zenith angle.  For each photolysis reaction 
        polynomial fits were created as a function of cosine(zenith 
        angle) at seven altitudes.  At a given altitude, the 
        photolysis frequency for each reaction is determined 
        by interpolation between bounding altitudes. Frequencies are
        valid from 0 to 12 km above mean sea level.
      *******************************************************************************/

      //begin interpolation
      //calculate rateparam1 (eg in fjmacra+fjmacrb, fjmacra is param1, often rateparam2 will be 0)
      if(paramNumID[0][i] >= 0) {
        amin0 = aparam[(amin-1)][paramNumID[0][i]];
        amin1 = aparam[(amin+6)][paramNumID[0][i]];
        amin2 = aparam[(amin+13)][paramNumID[0][i]];
        amin3 = aparam[(amin+20)][paramNumID[0][i]];
        amin4 = aparam[(amin+27)][paramNumID[0][i]];
        amin5 = aparam[(amin+34)][paramNumID[0][i]];
        amin6 = aparam[(amin+41)][paramNumID[0][i]];
        amax0 = aparam[(amax-1)][paramNumID[0][i]];
        amax1 = aparam[(amax+6)][paramNumID[0][i]];
        amax2 = aparam[(amax+13)][paramNumID[0][i]];
        amax3 = aparam[(amax+20)][paramNumID[0][i]];
        amax4 = aparam[(amax+27)][paramNumID[0][i]];
        amax5 = aparam[(amax+34)][paramNumID[0][i]];
        amax6 = aparam[(amax+41)][paramNumID[0][i]];

        zpolylow = amin0 + amin1*cz + amin2*cz2 + amin3*cz3 + amin4*cz4 + amin5*cz5 + amin6*cz6;
        zpolyhigh = amax0 + amax1*cz + amax2*cz2 + amax3*cz3 + amax4*cz4 + amax5*cz5 + amax6*cz6;

        rateparam1 = zfac1*zpolyhigh + zfac*zpolylow;
      }
      
      if(paramNumID[1][i] >= 0) {
      //calculate rateparam2
        amin0 = aparam[(amin-1)][paramNumID[1][i]];
        amin1 = aparam[(amin+6)][paramNumID[1][i]];
        amin2 = aparam[(amin+13)][paramNumID[1][i]];
        amin3 = aparam[(amin+20)][paramNumID[1][i]];
        amin4 = aparam[(amin+27)][paramNumID[1][i]];
        amin5 = aparam[(amin+34)][paramNumID[1][i]];
        amin6 = aparam[(amin+41)][paramNumID[1][i]];
        amax0 = aparam[(amax-1)][paramNumID[1][i]];
        amax1 = aparam[(amax+6)][paramNumID[1][i]];
        amax2 = aparam[(amax+13)][paramNumID[1][i]];
        amax3 = aparam[(amax+20)][paramNumID[1][i]];
        amax4 = aparam[(amax+27)][paramNumID[1][i]];
        amax5 = aparam[(amax+34)][paramNumID[1][i]];
        amax6 = aparam[(amax+41)][paramNumID[1][i]];

        zpolylow = amin0 + amin1*cz + amin2*cz2 + amin3*cz3 + amin4*cz4 + amin5*cz5 + amin6*cz6;
        zpolyhigh = amax0 + amax1*cz + amax2*cz2 + amax3*cz3 + amax4*cz4 + amax5*cz5 + amax6*cz6;

        rateparam2 = zfac1*zpolyhigh + zfac*zpolylow;
        
      }

      //bring it all together, calculate Rate using multipliers, rateparam1 and 2
      if(zenith >= 0 && zenith <= 1.57079632679) {
        Rate[i] = paramMult[0][i]*rateparam1 + paramMult[1][i]*rateparam2;
      } else {
        Rate[i] = 0.0;
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
      //printf("Before Euler, Species[%d]'s old abundance is Y: %e, updated with F+: %e, F-: %e, dt: %e\n", i, Y[i], FplusSum[i], FminusSum[i], dt);
			Y[i] = eulerUpdate(FplusSum[i], FminusSum[i], Yzero[i], dt);
      //printf("by Euler, Species[%d]'s new abundance is Y: %e, updated with F+: %e, F-: %e, dt: %e\n", i, Y[i], FplusSum[i], FminusSum[i], dt);
		}
	}
}

/* Checks for partial equilibrium between reaction groups */
void partialEquil(fern_real *Y, unsigned short numberReactions, int *ReacGroups, int **reactant, int **product, fern_real **final_k, int *pEquil, int *RGid, int numRG, fern_real tolerance, int eq)
{
	fern_real y_a;
	fern_real y_b;
	fern_real y_c;
	fern_real y_d;
	fern_real y_e;
	fern_real c1;
	fern_real c2;
	fern_real c3;
	fern_real c4;
	fern_real a;
	fern_real b;
	fern_real c;
	fern_real alpha;
	fern_real beta;
	fern_real gamma;
	fern_real q;
	fern_real y_eq_a;
	fern_real y_eq_b;
	fern_real y_eq_c;
	fern_real y_eq_d;
	fern_real y_eq_e;
	fern_real PE_val_a;
	fern_real PE_val_b;
	fern_real PE_val_c;
	fern_real PE_val_d;
	fern_real PE_val_e;
  int members;
	bool PEprintData = false;

		//final partial equilibrium loop for calculating equilibrium
		for(int i = 0; i < numRG; i++) {
          pEquil[i] = 0;
	        //reset RG reactant and product populations
            y_a = 0;
            y_b = 0;
            y_c = 0;
            y_d = 0;
            y_e = 0;
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
                    PE_val_a = fabs(y_a-y_eq_a)/(y_eq_a);
                    PE_val_b = fabs(y_b-y_eq_b)/(y_eq_b);
                    if(PE_val_a < tolerance && PE_val_b < tolerance) {
                        pEquil[i] = 1;
                    } 
					if(PEprintData)
						printf("RG %d: members=%d kf=%e kr=%e\n RGClass=%d c1=%f\n a=%e b=%e c=%e\n q=%e\n y0_eq: %e y0: %e y1_eq: %e y1: %e R0: %e R1: %e inEquilibrium: %d\n\n",i,members,final_k[0][i],final_k[1][i],ReacGroups[RGid[i]], c1, a, b, c, q, y_eq_a, y_a, y_eq_b, y_b, PE_val_a, PE_val_b, pEquil[i]);
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

                    PE_val_a = fabs(y_a-y_eq_a)/(y_eq_a);
                    PE_val_b = fabs(y_b-y_eq_b)/(y_eq_b);
                    PE_val_c = fabs(y_c-y_eq_c)/(y_eq_c);
                    if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance) {
                        pEquil[i] = 1;
                    } 
                    if(PEprintData)
						printf("RG %d: members=%d kf=%e kr=%e\n RGClass= %d c1=%f c2=%f\n a=%e b=%e c=%e\n q=%e\n y0_eq: %e y0: %e y1_eq: %e y1: %e y2_eq: %e y2: %e\nR0:%e R1: %e R2: %e inEquilibrium: %d\n\n",i, members,final_k[0][i],final_k[1][i],ReacGroups[RGid[i]], c1, c2, a, b, c, q, y_eq_a, y_a, y_eq_b, y_b, y_eq_c, y_c, PE_val_a, PE_val_b, PE_val_c, pEquil[i]);
                }
                else if(ReacGroups[RGid[i]] == 3) {
                    y_a = Y[reactant[0][RGid[i]]];
                    y_b = Y[reactant[1][RGid[i]]];
                    y_c = Y[reactant[2][RGid[i]]];
                    y_d = Y[product[0][RGid[i]]];
                    c1 = y_a-y_b;
                    c2 = y_a-y_c;
                    c3 = ((1/3)*(y_a+y_b+y_c))+y_d;
                    a = final_k[0][i]*(c1+c2)-final_k[0][i]*y_a;
                    b = -((final_k[0][i]*c1*c2)+final_k[1][i]);
                    c = final_k[1][i]*(c3+(c1/3)+(c2/3));
                    q = (4*a*c)-(b*b);

                    y_eq_a = ((-.5/a)*(b+sqrt(-q)));
                    y_eq_b = y_eq_a-c1;
                    y_eq_c = y_eq_a-c2;
                    y_eq_d = c3-y_eq_a+((1/3)*(c1+c2));

                    PE_val_a = fabs(y_a-y_eq_a)/(y_eq_a);
                    PE_val_b = fabs(y_b-y_eq_b)/(y_eq_b);
                    PE_val_c = fabs(y_c-y_eq_c)/(y_eq_c);
                    PE_val_d = fabs(y_d-y_eq_d)/(y_eq_d);
                    if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance && PE_val_d < tolerance) {
                        pEquil[i] = 1;
                    } 
                    if(PEprintData)
						printf("RG %d: members=%d kf=%e kr=%e\n RGClass= %d c1=%f c2=%f c3=%f\n a=%e b=%e c=%e\n q=%e\n y0_eq: %e y0: %e y1_eq: %e y1: %e y2_eq: %e y2:%e y3_eq: %e y3: %e\nR0: %e R1: %e R2: %e R3: %e, inEquilibrium: %d\n\n",i, members,final_k[0][i],final_k[1][i],ReacGroups[RGid[i]], c1, c2, c3, a, b, c, q, y_eq_a, y_a, y_eq_b, y_b, y_eq_c, y_c, y_eq_d, y_d, PE_val_a, PE_val_b, PE_val_c, PE_val_d, pEquil[i]);
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
				
					PE_val_a = fabs(y_a-y_eq_a)/(y_eq_a);
					PE_val_b = fabs(y_b-y_eq_b)/(y_eq_b);
					PE_val_c = fabs(y_c-y_eq_c)/(y_eq_c);
					PE_val_d = fabs(y_d-y_eq_d)/(y_eq_d);
					if(PE_val_a > tolerance && PE_val_b < tolerance && PE_val_c < tolerance && PE_val_d < tolerance) {
						pEquil[i] = 1;
					}
                    if(PEprintData)
						printf("RG %d: members=%d kf=%e kr=%e\n RGClass= %d c1=%f c2=%f c3=%f\n a=%e b=%e c=%e\n q=%e\n y0_eq: %e y0: %e y1_eq: %e y1: %e y2_eq: %e y2: %e y3_eq: %e y3: %e\nR0: %e R1: %e R2: %e R3: %e inEquilibrium: %d\n\n",i, members,final_k[0][i],final_k[1][i],ReacGroups[RGid[i]], c1, c2, c3, a, b, c, q, y_eq_a, y_a, y_eq_b, y_b, y_eq_c, y_c, y_eq_d, y_d, PE_val_a, PE_val_b, PE_val_c, PE_val_d, pEquil[i]);
				}
				else if(ReacGroups[RGid[i]] == 5) {
                    y_a = Y[reactant[0][RGid[i]]];
                    y_b = Y[reactant[1][RGid[i]]];
                    y_c = Y[product[0][RGid[i]]];
                    y_d = Y[product[1][RGid[i]]];
					y_e = Y[product[2][RGid[i]]];

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

                    y_eq_a = ((-.5/a)*(b+sqrt(-q)));
                    y_eq_b = y_eq_a-c2;
                    y_eq_c = alpha-y_eq_a;
                    y_eq_d = beta-y_eq_a;
                    y_eq_e = gamma-y_eq_a;

                    PE_val_a = fabs(y_a-y_eq_a)/(y_eq_a);
                    PE_val_b = fabs(y_b-y_eq_b)/(y_eq_b);
                    PE_val_c = fabs(y_c-y_eq_c)/(y_eq_c);
                    PE_val_d = fabs(y_d-y_eq_d)/(y_eq_d);
                    PE_val_e = fabs(y_e-y_eq_e)/(y_eq_e);
                    if(PE_val_a < tolerance && PE_val_b < tolerance && PE_val_c < tolerance && PE_val_d < tolerance && PE_val_e < tolerance) {
                        pEquil[i] = 1;
                    } 
                    if(PEprintData)
						printf("RG %d: members=%d kf=%e kr=%e\n RGClass= %d c1=%f c2=%f c3=%f c4=%f\n a=%e b=%e c=%e\n q=%e\n y0_eq: %e y0: %e y1_eq: %e y1: %e y2_eq: %e y2: %e y3_eq: %e y3: %e y4_eq: %e y4: %e\nR0: %e R1: %e R2: %e R3: %e R4:%e inEquilibrium: %d\n\n",i, members,final_k[0][i],final_k[1][i],ReacGroups[RGid[i]], c1, c2, c3, c4, a, b, c, q, y_eq_a, y_a, y_eq_b, y_b, y_eq_c, y_c, y_eq_d, y_d, y_eq_e, y_e, PE_val_a, PE_val_b, PE_val_c, PE_val_d, PE_val_e, pEquil[i]);
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
