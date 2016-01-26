void *init_fern();
void integrate_fern(void *f, fern_real dt, fern_real tmp, fern_real rho,
	fern_real *xIn, fern_real *xOut);

typedef struct {
  Network *reacNetwork;
  IntegrationData *integrationData;
  Globals *globals;
  fire::IStepper *stepper;
} FernData;

void loadParameters(Network & network, IntegrationData * data,
        const char * filename);
