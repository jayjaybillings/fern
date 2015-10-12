// ------------------------------------------------------------------------------------------------
//  Class to illustrate reading in objects originally serialized using
//  the class ReactionClass1.  The input filename is specified as
//  a command-line argument.  That is, invoke with
//
//      java LoadR1 fileName
//
//  where fileName is the name of the file to be deserialized
//  that was originally serialized with ReactionClass1.  This class
//  reads in all ReactionClass1 objects in the file specified and
//  prints out all data fields for each object.
// -------------------------------------------------------------------------------------------------


import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;

class LoadR1 {

    public static void main(String[] args) {

        GraphicsGoodies2 gg = new GraphicsGoodies2();

        if (args.length != 1) {      // Check for fileName argument
            System.err.println();
            System.err.println("Usage: java LoadR1 <file>");
            System.err.println();
            System.exit(1);
        }

        // Create an input object stream wrapped around a file input
        // stream.  Then read the initial Integer stored giving the
        // number of objects that were serialized.  Then
        // read the objects in sequentially, first
        // casting to the correct type (the return type of readObject()
        // is Object, so we have to cast this to the ReactionClass1
        // type that we expect) and then assigning to elements
        // of the ReactionClass1 array instance[].

        try {

            // Take the name of the input file from standard input;
            // wrap this file in an object input stream

            FileInputStream fileIn = new FileInputStream(args[0]);
            ObjectInputStream in = new ObjectInputStream(fileIn);

            // Read from the input stream the initial integer giving
            // the number of objects that were serialized in this file

            int numberObjects = in.readInt();
            System.out.println();
            System.out.println("numberObjects="+numberObjects);
            System.out.println();

            // Read from the input stream the 9-d int array giving
            // the number of reactions of each type.  Entry 0 is
            // the total (numberObjects).  Array entries 1-8 give the
            // the subtotals for each of the 8 reaction types

            int [] numberEachType = (int []) in.readObject();
            System.out.println("Number of Reactions of All Types = "
                + numberEachType[0]);
            for (int k=1; k<9; k++) {
                System.out.println("Number Reactions of Type "
                     + k + " = " + numberEachType[k]);
            }
            System.out.println();

            // Create an array to hold the reaction objects as they are
            // deserialized

            ReactionClass1 [] instance = new ReactionClass1 [numberObjects];

            // Deserialize the objects to the array instance[]

            int i=0;
            while (i < numberObjects) {
                instance[i] = (ReactionClass1)in.readObject();
                i++;
            }

            // Close the input streams

            in.close();
            fileIn.close();

            // Print out the instance variables from the
            // serialized objects that we have just read

            for (int j=0; j < instance.length; j++) {
                System.out.println("Serial index: "+j+"   " +instance[j].reacString
                    + "       Ref: "+instance[j].refString);
                System.out.println("Reaction Index = "+instance[j].reacIndex + " numberReactants = "
                    + instance[j].numberReactants + " numberProducts = "
                    + instance[j].numberProducts +"  Q = "+instance[j].Q);
                System.out.println("reverseR = " + instance[j].reverseR
                    + "  resonant = " + instance[j].resonant
                    + "  nonResonant = " + instance[j].nonResonant
                    + "  ecFlag = " + instance[j].ecFlag);

                String temp = "";
                for (int k=0; k<instance[j].numberReactants; k++) {
                    temp += "Zin(" + (k+1) + ")=" + instance[j].isoIn[k].x;
                    temp += "  Nin(" + (k+1) + ")="+ instance[j].isoIn[k].y + "  ";
                }
                System.out.println(temp);

                temp = "";
                for (int k=0; k<instance[j].numberProducts; k++) {
                    temp += "Zout(" + (k+1) + ")=" + instance[j].isoOut[k].x;
                    temp += "  Nout(" + (k+1) + ")=" + instance[j].isoOut[k].y + "  ";
                }
                System.out.println(temp);

                System.out.println("p1 = "+instance[j].p0
                    + " p2 = "+instance[j].p1
                    + " p3 = "+instance[j].p2
                    + " p4 = "+instance[j].p3);

                System.out.println("p5 = "+instance[j].p4
                    + " p6 = "+instance[j].p5
                    + " p7 = "+instance[j].p6);

                // Following changes in rate printing required because
                // with new faster rate calculation with powers and logs of T computed once per
                // timestep, need to specify these quantities in ReactionClass1 before calling the
                // .rate method (As of svn revision 397, Feb. 18, 2009).

                String tss = "Rate(T9): ";
                double t9 = 0.01;
                ReactionClass1.logT9 = Math.log(t9);
                ReactionClass1.T913 = Math.pow(t9,0.3333333);
                ReactionClass1.T953 = Math.pow(t9,1.6666666);
                tss += (" R("+t9+")="+gg.decimalPlace(2,instance[j].rate(t9)));
                t9 = 0.1;
                ReactionClass1.logT9 = Math.log(t9);
                ReactionClass1.T913 = Math.pow(t9,0.3333333);
                ReactionClass1.T953 = Math.pow(t9,1.6666666);
                tss += (" R("+t9+")="+gg.decimalPlace(2,instance[j].rate(t9)));
                t9=1.0;
                ReactionClass1.logT9 = Math.log(t9);
                ReactionClass1.T913 = Math.pow(t9,0.3333333);
                ReactionClass1.T953 = Math.pow(t9,1.6666666);
                tss += (" R("+t9+")="+gg.decimalPlace(2,instance[j].rate(t9)));
                t9=10.0;
                ReactionClass1.logT9 = Math.log(t9);
                ReactionClass1.T913 = Math.pow(t9,0.3333333);
                ReactionClass1.T953 = Math.pow(t9,1.6666666);
                tss += (" R("+t9+")="+gg.decimalPlace(2,instance[j].rate(t9)));
                
                System.out.println(tss);
                System.out.println();
            }

        }
        catch (Exception e) {
            System.out.println(e);
        }
    }

}  /* End class LoadR1 */