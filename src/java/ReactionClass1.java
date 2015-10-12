// -------------------------------------------------------------------------------------------------------
// The class ReactionClass1 creates objects that hold the relevant
// information required to calculate rates for a particular reaction
// channel of an isotope.
// -------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;

public class ReactionClass1 implements Serializable {

    int reacIndex, numberReactants, numberProducts;
    String reacString, refString;
    boolean ecFlag, reverseR, resonant, nonResonant;
    Point [] isoIn = new Point[3];    // Entrance channel Z,N pairs
    Point [] isoOut = new Point[4];  // Exit channel Z,N pairs
    double Q;
    double p0, p1, p2, p3, p4, p5, p6;
    double prefac;
    private final double THIRD = 0.33333333333333333333;
    private final double FIVETHIRDS = 1.666666666666666666666;

    static double logT9;
    static double T913;
    static double T953;


    // ---------------------------------------------------------------
    // Constructor
    // ---------------------------------------------------------------

    ReactionClass1 (int reacIndex, int numberReactants,
        int numberProducts, boolean ecFlag,
        boolean reverseR, boolean resonant,
        boolean nonResonant, String reacString,
        String refString, Point [] isoIn,
        Point [] isoOut, double Q, double [] parms) {

       // Copy argument list into instance variables

       this.reacIndex = reacIndex;
       this.numberReactants = numberReactants;
       this.numberProducts = numberProducts;
       this.ecFlag = ecFlag;
       this.reverseR = reverseR;
       this.resonant = resonant;
       this.nonResonant = nonResonant;
       this.reacString = reacString;
       this.refString = refString;
       for (int i=0; i<=2; i++) {
           this.isoIn[i] = isoIn[i];
           this.isoOut[i] = isoOut[i];
       }
       this.isoOut[3] = isoOut[3];
       this.Q = Q;
       this.p0 = parms[0];
       this.p1 = parms[1];
       this.p2 = parms[2];
       this.p3 = parms[3];
       this.p4 = parms[4];
       this.p5 = parms[5];
       this.p6 = parms[6];

       setprefac();

    }


    // --------------------------------------------------------
    //  Method to compute statistical factor
    // --------------------------------------------------------

    void setprefac() {

        prefac=1.0;

        if( reacIndex > 3 && reacIndex < 8) {
            if(isoIn[0].x == isoIn[1].x && isoIn[0].y == isoIn[1].y){
                prefac=0.5;
            }
        }

        if (reacIndex == 8) {
            if( isoIn[0].x == isoIn[1].x
                    && isoIn[1].x == isoIn[2].x
                    && isoIn[0].y == isoIn[1].y
                    && isoIn[1].y == isoIn[2].y) {
                prefac=0.166666666667;
            }
        }
    }



    // ---------------------------------------------------------------------------------------
    //  Method to return the intrinsic rate as a function of the
    //  temperature.   The factors like T913 are precomputed to
    //  increase speed rather than recomputing each call.
    // ---------------------------------------------------------------------------------------

    double rate (double T9) {

        return Math.exp( p0 + p1/T9 + p2/T913 + p3*T913 + p4*T9 + p5*T953 + p6*logT9 );

    }


    // -------------------------------------------------------------------------------
    //  Method to calculate full rate from temperature,
    //  density, and electron fraction (if electron capture
    //  plays role, latter is needed).
    // -------------------------------------------------------------------------------

    double prob(double T9, double rho, double Ye) {

        double fac = prefac;

        if(reacIndex > 3) {
            fac *= (rho*StochasticElements.Y[isoIn[0].x][isoIn[0].y]);
        }
        if(ecFlag) fac *= (rho*Ye);
        if(reacIndex == 8) {
            fac *= (rho*StochasticElements.Y[isoIn[1].x][isoIn[1].y]);
        }

        return fac*rate(T9);

    }
    
    
    // ---------------------------------------------------------------------------------------------
    //  Method to calculate effective rate constant from temperature,
    //  density, and electron fraction (if electron capture
    //  plays role, latter is needed).  This differs from the method
    //  prob( ) in that it doesn't multiply by the abundance factors Y,
    //  so is better suited for implementing partial equilibrium methods.
    // ---------------------------------------------------------------------------------------------
    
    double returnk(double T9, double rho, double Ye) {

        double fac = prefac;
        
        if(reacIndex > 3) {
            fac *= (rho);
        }
        if(ecFlag) fac *= (rho*Ye);
        if(reacIndex == 8) {
            fac *= (rho);
        }

        return fac*rate(T9);

    }



    // ---------------------------------------------------------------------------------------------------------------------
    //  Method to update N and Z plane entries and total E released.  Overloaded;
    //  this version assumes one transition at a time and accepts no argument.
    //  The second version accepts an integer argument for the number of transitions
    //  to implement at one time. This version is generally no longer used.
    // ----------------------------------------------------------------------------------------------------------------------

    void newZNQ() {

        byte Z = StochasticElements.Z;
        byte N = StochasticElements.N;

        switch(this.reacIndex) {

            case 0:

                break;

            case 1:

                StochasticElements.pop[Z][N] --;
                StochasticElements.dERelease += this.Q;
                StochasticElements.Y[Z][N] =
                    StochasticElements.pop[Z][N]/StochasticElements.nT;
                StochasticElements.Z = (byte)isoOut[0].x;
                StochasticElements.N = (byte)isoOut[0].y;
                StochasticElements.pop[isoOut[0].x][isoOut[0].y] ++;
                StochasticElements.Y[isoOut[0].x][isoOut[0].y] =
                    StochasticElements.pop[isoOut[0].x][isoOut[0].y]
                    / StochasticElements.nT;
                break;

            case 2:

                StochasticElements.pop[Z][N] --;
                StochasticElements.dERelease += this.Q;
                StochasticElements.Y[Z][N] =
                    StochasticElements.pop[Z][N]/StochasticElements.nT;
                StochasticElements.Z = (byte)isoOut[1].x;
                StochasticElements.N = (byte)isoOut[1].y;
                StochasticElements.pop[isoOut[0].x][isoOut[0].y] ++;
                StochasticElements.Y[isoOut[0].x][isoOut[0].y] =
                    StochasticElements.pop[isoOut[0].x][isoOut[0].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[1].x][isoOut[1].y] ++;
                StochasticElements.Y[isoOut[1].x][isoOut[1].y] =
                    StochasticElements.pop[isoOut[1].x][isoOut[1].y]
                    / StochasticElements.nT;
                break;

            case 3:

                StochasticElements.pop[Z][N] --;
                StochasticElements.dERelease += this.Q;
                StochasticElements.Y[Z][N] =
                    StochasticElements.pop[Z][N]/StochasticElements.nT;
                StochasticElements.Z = (byte)isoOut[2].x;
                StochasticElements.N = (byte)isoOut[2].y;
                StochasticElements.pop[isoOut[0].x][isoOut[0].y] ++;
                StochasticElements.Y[isoOut[0].x][isoOut[0].y] =
                    StochasticElements.pop[isoOut[0].x][isoOut[0].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[1].x][isoOut[1].y] ++;
                StochasticElements.Y[isoOut[1].x][isoOut[1].y] =
                    StochasticElements.pop[isoOut[1].x][isoOut[1].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[2].x][isoOut[2].y] ++;
                StochasticElements.Y[isoOut[2].x][isoOut[2].y] =
                    StochasticElements.pop[isoOut[2].x][isoOut[2].y]
                    / StochasticElements.nT;
                break;

            case 4:

                StochasticElements.pop[Z][N] --;
                StochasticElements.dERelease += this.Q;
                StochasticElements.Y[Z][N] =
                    StochasticElements.pop[Z][N]/StochasticElements.nT;
                StochasticElements.Z = (byte)isoOut[0].x;
                StochasticElements.N = (byte)isoOut[0].y;
                StochasticElements.pop[isoIn[0].x][isoIn[0].y] --;
                StochasticElements.Y[isoIn[0].x][isoIn[0].y] =
                    StochasticElements.pop[isoIn[0].x][isoIn[0].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[0].x][isoOut[0].y] ++;
                StochasticElements.Y[isoOut[0].x][isoOut[0].y] =
                    StochasticElements.pop[isoOut[0].x][isoOut[0].y]
                    / StochasticElements.nT;
                break;

            case 5:

                StochasticElements.pop[Z][N] --;
                StochasticElements.dERelease += this.Q;
                StochasticElements.Y[Z][N] =
                    StochasticElements.pop[Z][N]/StochasticElements.nT;
                StochasticElements.Z = (byte)isoOut[1].x;
                StochasticElements.N = (byte)isoOut[1].y;
                StochasticElements.pop[isoIn[0].x][isoIn[0].y] --;
                StochasticElements.Y[isoIn[0].x][isoIn[0].y] =
                    StochasticElements.pop[isoIn[0].x][isoIn[0].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[0].x][isoOut[0].y] ++;
                StochasticElements.Y[isoOut[0].x][isoOut[0].y] =
                    StochasticElements.pop[isoOut[0].x][isoOut[0].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[1].x][isoOut[1].y] ++;
                StochasticElements.Y[isoOut[1].x][isoOut[1].y] =
                    StochasticElements.pop[isoOut[1].x][isoOut[1].y]
                    / StochasticElements.nT;
                break;

            case 6:

                StochasticElements.pop[Z][N] --;
                StochasticElements.dERelease += this.Q;
                StochasticElements.Y[Z][N] =
                    StochasticElements.pop[Z][N]/StochasticElements.nT;
                StochasticElements.Z = (byte)isoOut[2].x;
                StochasticElements.N = (byte)isoOut[2].y;
                StochasticElements.pop[isoIn[0].x][isoIn[0].y] --;
                StochasticElements.Y[isoIn[0].x][isoIn[0].y] =
                    StochasticElements.pop[isoIn[0].x][isoIn[0].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[0].x][isoOut[0].y] ++;
                StochasticElements.Y[isoOut[0].x][isoOut[0].y] =
                    StochasticElements.pop[isoOut[0].x][isoOut[0].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[1].x][isoOut[1].y] ++;
                StochasticElements.Y[isoOut[1].x][isoOut[1].y] =
                    StochasticElements.pop[isoOut[1].x][isoOut[1].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[2].x][isoOut[2].y] ++;
                StochasticElements.Y[isoOut[2].x][isoOut[2].y] =
                    StochasticElements.pop[isoOut[2].x][isoOut[2].y]
                    / StochasticElements.nT;
                break;

            case 7:

                StochasticElements.pop[Z][N] --;
                StochasticElements.dERelease += this.Q;
                StochasticElements.Y[Z][N] =
                    StochasticElements.pop[Z][N]/StochasticElements.nT;
                StochasticElements.Z = (byte)isoOut[3].x;
                StochasticElements.N = (byte)isoOut[3].y;
                StochasticElements.pop[isoIn[0].x][isoIn[0].y] --;
                StochasticElements.Y[isoIn[0].x][isoIn[0].y] =
                    StochasticElements.pop[isoIn[0].x][isoIn[0].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[0].x][isoOut[0].y] ++;
                StochasticElements.Y[isoOut[0].x][isoOut[0].y] =
                    StochasticElements.pop[isoOut[0].x][isoOut[0].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[1].x][isoOut[1].y] ++;
                StochasticElements.Y[isoOut[1].x][isoOut[1].y] =
                    StochasticElements.pop[isoOut[1].x][isoOut[1].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[2].x][isoOut[2].y] ++;
                StochasticElements.Y[isoOut[2].x][isoOut[2].y] =
                    StochasticElements.pop[isoOut[2].x][isoOut[2].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[3].x][isoOut[3].y] ++;
                StochasticElements.Y[isoOut[3].x][isoOut[3].y] =
                    StochasticElements.pop[isoOut[3].x][isoOut[3].y]
                    / StochasticElements.nT;
                break;

            case 8:

                StochasticElements.pop[Z][N] --;
                StochasticElements.dERelease += this.Q;
                StochasticElements.Y[Z][N] =
                    StochasticElements.pop[Z][N]/StochasticElements.nT;
                StochasticElements.Z = (byte)isoOut[0].x;
                StochasticElements.N = (byte)isoOut[0].y;
                StochasticElements.pop[isoIn[0].x][isoIn[0].y] --;
                StochasticElements.Y[isoIn[0].x][isoIn[0].y] =
                    StochasticElements.pop[isoIn[0].x][isoIn[0].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoIn[1].x][isoIn[1].y] --;
                StochasticElements.Y[isoIn[1].x][isoIn[1].y] =
                    StochasticElements.pop[isoIn[1].x][isoIn[1].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[0].x][isoOut[0].y] ++;
                StochasticElements.Y[isoOut[0].x][isoOut[0].y] =
                    StochasticElements.pop[isoOut[0].x][isoOut[0].y]
                    / StochasticElements.nT;
                StochasticElements.pop[isoOut[1].x][isoOut[1].y] ++;
                StochasticElements.Y[isoOut[1].x][isoOut[1].y] =
                    StochasticElements.pop[isoOut[1].x][isoOut[1].y]
                    / StochasticElements.nT;
                break;
        }
    }


    // ------------------------------------------------------------------------------------------------
    //  Second (overloaded) method to update N and Z plane entries
    //  and total E released.  This one accepts an integer argument that
    //  specifies the number of transitions to implement at once.  It is
    //  the one now used exclusively.
    // --------------------------------------------------------------------------------------------------

    void newZNQ(double popOut) {

        byte Z = StochasticElements.Z;
        byte N = StochasticElements.N;

        switch(this.reacIndex) {

            case 0:

                break;

            case 1:        // a -> b

                StochasticElements.dpopMinus[Z][N] += popOut;
                StochasticElements.dERelease += this.Q*popOut;
                StochasticElements.dpopPlus[isoOut[0].x][isoOut[0].y] += popOut;

                break;

            case 2:        // a -> b + c

                StochasticElements.dpopMinus[Z][N] += popOut;
                StochasticElements.dERelease += this.Q*popOut;
                StochasticElements.dpopPlus[isoOut[0].x][isoOut[0].y] += popOut;
                StochasticElements.dpopPlus[isoOut[1].x][isoOut[1].y] += popOut;

                break;

            case 3:        // a -> b + c + d

                StochasticElements.dpopMinus[Z][N] += popOut;
                StochasticElements.dERelease += this.Q*popOut;
                StochasticElements.dpopPlus[isoOut[0].x][isoOut[0].y] += popOut;
                StochasticElements.dpopPlus[isoOut[1].x][isoOut[1].y] += popOut;
                StochasticElements.dpopPlus[isoOut[2].x][isoOut[2].y] += popOut;

                break;

            case 4:        // a + b -> c

                StochasticElements.dpopMinus[Z][N] += popOut;
                StochasticElements.dERelease += this.Q*popOut;
                StochasticElements.dpopMinus[isoIn[0].x][isoIn[0].y] += popOut;
                StochasticElements.dpopPlus[isoOut[0].x][isoOut[0].y] += popOut;

                break;

            case 5:        // a + b -> c + d

                StochasticElements.dpopMinus[Z][N] += popOut;
                StochasticElements.dERelease += this.Q*popOut;
                StochasticElements.dpopMinus[isoIn[0].x][isoIn[0].y] += popOut;
                StochasticElements.dpopPlus[isoOut[0].x][isoOut[0].y] += popOut;
                StochasticElements.dpopPlus[isoOut[1].x][isoOut[1].y] += popOut;

                break;

            case 6:        // a + b -> c + d + e

                StochasticElements.dpopMinus[Z][N] += popOut;
                StochasticElements.dERelease += this.Q*popOut;
                StochasticElements.dpopMinus[isoIn[0].x][isoIn[0].y] += popOut;
                StochasticElements.dpopPlus[isoOut[0].x][isoOut[0].y] += popOut;
                StochasticElements.dpopPlus[isoOut[1].x][isoOut[1].y] += popOut;
                StochasticElements.dpopPlus[isoOut[2].x][isoOut[2].y] += popOut;

                break;

            case 7:        // a + b -> c + d + e + f

                StochasticElements.dpopMinus[Z][N] += popOut;
                StochasticElements.dERelease += this.Q*popOut;
                StochasticElements.dpopMinus[isoIn[0].x][isoIn[0].y] += popOut;
                StochasticElements.dpopPlus[isoOut[0].x][isoOut[0].y] += popOut;
                StochasticElements.dpopPlus[isoOut[1].x][isoOut[1].y] += popOut;
                StochasticElements.dpopPlus[isoOut[2].x][isoOut[2].y] += popOut;
                StochasticElements.dpopPlus[isoOut[3].x][isoOut[3].y] += popOut;

                break;

            case 8:        // a + b + c -> d (+e)

                StochasticElements.dpopMinus[Z][N] += popOut;
                StochasticElements.dERelease += this.Q*popOut;
                StochasticElements.dpopMinus[isoIn[0].x][isoIn[0].y] += popOut;
                StochasticElements.dpopMinus[isoIn[1].x][isoIn[1].y] += popOut;
                StochasticElements.dpopPlus[isoOut[0].x][isoOut[0].y] += popOut;

                if(this.numberProducts>1){
                //if(isoOut[1].x > 0){
                        StochasticElements.dpopPlus[isoOut[1].x][isoOut[1].y] += popOut;
                }
				
                break;

        }

    }


    // ---------------------------------------------------------------------------------------------------
    //  Method to serialize ReactionClass1 objects to disk output files.
    //  It takes three arguments:  the first is the filename, the second
    //  is a 9-d array holding the number of reactions being stored of
    //  each reaction class.  Entry 0 of this array is the total number
    //  of reactions, while entries 1-8 are the subtotals for each
    //  of the 8 reaction types.  The third argument is an
    //  array of objects to serialize to that file. (See the class
    //  loadR1 for an example of how to read these serialized objects
    //  back in from the disk file.)  The method first writes an int
    //  to the output stream giving the number of objects to be serialized
    //  in this file.  This is followed by the array holding the number
    //  of reactions of each type (written as an object).  Then the
    //  reaction objects are written sequentially to the file.
    // ------------------------------------------------------------------------------------------------------

    static void serializeIt (String fileName, int [] numberEachType,
        ReactionClass1 [] objectName) {

        try {
            FileOutputStream fileOut = new FileOutputStream(fileName);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);

            // First write to the output stream an integer giving the
            // number of objects that are to be serialized in this file

            out.writeInt(objectName.length);

            // Then write the 9-d array holding the number of reactions of
            // each type.  Note: entry 0 of this array is the same as
            // the int written in the previous step.

            out.writeObject(numberEachType);

            // Then, serialize the reaction objects contained in the
            // array objectName.

            for (int i=0; i<objectName.length; i++) {
                out.writeObject(objectName[i]);
            }

        }
        catch (Exception e) {
            System.out.println(e);
        }
    }

}  /*  End class ReactionClass1  */
