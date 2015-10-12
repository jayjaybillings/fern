// -------------------------------------------------------------------------------------------------------
//  Class to hold Solar elemental abundances in static array sab[Z][N]
// -------------------------------------------------------------------------------------------------------

class SolarAbundances {

    static int z = 92;
    static int n = 150;
    static double [][] sab = new double[z][n];

    // Static code block initialization

    static {

        // Non-zero Solar abundances:

        sab[1][0] = 0.70693;      // hydrogen-1
        sab[2][2] = 0.06884;      // helium-4
        sab[6][6] = 2.5283E-4;   // carbon-12
        sab[6][7] = 2.8114E-6;   // carbon-13
        sab[7][7] = 7.8675E-5;   // nitrogen-14
        sab[7][8] = 2.910E-7;    // nitrogen-15
        sab[8][8] = 6.002E-4;    // oxygen-16
        sab[8][9] = 2.910E-7;    // oxygen-17
        sab[8][10] = 1.207E-6;   // oxygen-18
        sab[10][10] = 7.8E-5;    // neon-20

    }

}