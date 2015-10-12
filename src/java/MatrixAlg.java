// -----------------------------------------------------------------------------------------
//  Class containing methods to deal with vectors and matrices,
//  Test by executing java MatrixAlg.
// -----------------------------------------------------------------------------------------

public class MatrixAlg {

    // Main method to test the matrix algebra methods.  Because the matrix
    // algebra methods can throw the homegrown exception MatrixAlgebraException,
    // any method using them must either declare this exception in a
    // throws clause of the method declaration like:
    //
    //      void myMethod () throws MatrixAlgebraException {
    //           ...
    //      }
    //
    // or it must be caught in a try-catch clause pair.  In the examples
    // below we illustrate both.  That is, the main method is declared
    // to throw the exception MatrixAlgebraException, which satisfies the syntax
    // requirement, but in addition in some cases we will handle the exception
    // explicitly with a try-catch clause, and in others we will let the exception
    // propagate up the stack to be handled by calling routines.  To see how these
    // different ways of handling the exceptions work, try changing the number
    // of components in one of the vectors to force the exception to be thrown
    // in each example.


    public static void main(String[] args) throws MatrixAlgebraException{

        // Test integer version of vector scalar product.  In this case
        // we catch any MatrixAlgebraException exceptions explicitly.  Test
        // the exception handling by making the lengths of vi1 and vi2 different.

        int [] vi1 = {2,3,4,5};
        int [] vi2 = {1,2,3,4};
        System.out.println();

        try {

            System.out.println("(int[],int[]) scalar product = "
                                   + scalarProd(vi1,vi2));
        }
        catch (MatrixAlgebraException e) {
            System.out.println(
                "\nAbort: Vectors not of same length in scalar product!"
                +"  Exception handled\nexplicitly at this level"
                + " and execution continues.\n") ;

           // Add any additional statements to tell the program
           // what to do about this exception.  If no others are added,
           // the above message is printed and execution continues.
        }


        // Test double version of vector scalar product.  In this case
        // we don't catch any MatrixAlgebraException exceptions
        // explicitly.  Thus, they propagate up the stack and eventually
        // halt execution.  Test the exception handling by making the
        // lengths of vd1 and vd2 different.

        double [] vd1 = {2.0, 4.0, 4.0, 3.0, 2.0};
        double [] vd2 = {1.0, 2.0, 3.0, 4.0, 12.0};
        System.out.println("(double[],double[]) scalar product = " + scalarProd(vd1,vd2));


        // Test mixed int-double version of vector scalar product
        // that returns a double.  In this case
        // we don't catch any MatrixAlgebraException exceptions
        // explicitly.  Thus, they propagate up the stack and eventually
        // halt execution.  Test this by making the lengths of vi3 and
        // vd3 different.

        int [] vi3 = {2, 3, 4};
        double [] vd3 = {1.5, 3.5, 3.0};
        double temp = scalarProd(vi3,vd3);
        System.out.println("(double) (int[],double[]) scalar product = " + temp);

    }  /* End method main */


    // Define 3 overloaded constructors that allow respectively the
    // two vector arguments to be (int[], int[]), returning int,
    // (double[], double[]), returning double, or (int[], double[]),
    // returning double.


    // --------------------------------------------------------------------------------------------
    //  Integer version of overloaded scalar product method.
    //  Declared to throw homegrown MatrixAlgebraException if
    //  the lengths of the two vectors are not the same.
    //  "throws MatrixAlgebraException" must either be declared
    //  in a throws clause, or this exception must be handled
    //  with try-catch clauses, in any methods calling this one
    //  (see main method for examples of both).  Method is
    //  static, so it can be invoked directly from the class
    //  without having to instantiate: Matrixalg.scalarProd(v1,v2);
    // ---------------------------------------------------------------------------------------------

    public static int scalarProd (int[] vector1, int[] vector2) throws MatrixAlgebraException {
        int prod = 0;
        String eString =
            "\nIncompatible lengths: int vector1 length = "
            + vector1.length + "  int vector2 length = "
            + vector2.length;

        //if (vector1.length != vector2.length) {exitError(eString);}
        if (vector1.length != vector2.length) {
            throw new MatrixAlgebraException(eString);
        } else {
            for (int i=0; i<vector1.length; i++) {
                prod += vector1[i] * vector2[i];
            }
        }
        return prod;
    }


    // -------------------------------------------------------------------------------------------
    //  The double version of overloaded scalar product method.
    //  Declared to throw homegrown MatrixAlgebraException if
    //  the lengths of the two vectors are not the same.
    //  "throws MatrixAlgebraException" must either be declared
    //  in a throws clause, or this exception must be handled
    //  with try-catch clauses, in any methods calling this one
    //  (see main method for examples of both).  Method is
    //  static, so it can be invoked directly from the class
    //  without having to instantiate: Matrixalg.scalarProd(v1,v2);
    // --------------------------------------------------------------------------------------------

    public static double scalarProd (double[] vector1, double[] vector2) throws MatrixAlgebraException {
        double prod = 0;
        String eString =
           "\nIncompatible lengths: double[] vector1 length = "
            + vector1.length + "  double[] vector2 length = "
            + vector2.length;

        if (vector1.length != vector2.length) {
            throw new MatrixAlgebraException(eString);
        } else {
            for (int i=0; i<vector1.length; i++) {
                prod += vector1[i] * vector2[i];
            }
        }
        return prod;
    }


    // -------------------------------------------------------------------------------------------
    //  The mixed int-double version of overloaded scalar product
    //  method that returns the scalar product as a double.
    //  Declared to throw homegrown MatrixAlgebraException if
    //  the lengths of the two vectors are not the same.
    //  "throws MatrixAlgebraException" must either be declared
    //  in a throws clause, or this exception must be handled
    //  with try-catch clauses, in any methods calling this one
    //  (see main method for examples of both).  Method is
    //  static, so it can be invoked directly from the class
    //  without having to instantiate: Matrixalg.scalarProd(v1,v2);
    // --------------------------------------------------------------------------------------------

    public static double scalarProd (int [] vector1, double[] vector2) throws MatrixAlgebraException {
        double prod = 0;
        String eString =
           "\nIncompatible lengths: int[] vector1 length = "
            + vector1.length + "  double[] vector2 length = "
            + vector2.length;

        if (vector1.length != vector2.length) {
            throw new MatrixAlgebraException(eString);
        } else {
            for (int i=0; i<vector1.length; i++) {
                prod += vector1[i] * vector2[i];
            }
        }
        return prod;
    }


}  /* End class MatrixAlg */


// -----------------------------------------------------------------------------------
//  Define a specialized exception to be thrown if we
//  attempt matrix operations that are illegal (such as
//  trying to take scalar product of two vectors not of
//  the same length).  In this example, the constructors
//  just invoke the contructors of the superclass Exception.
//  To invoke in a method, insert
//
//         throw new MatrixAlgebraException(s);
//
//  where s is a string that will be displayed when the
//  exception is thrown (unless it is caught and handled
//  explicitly).  Insertion of this statement requires
//  either that the exception be handled in try-catch
//  clauses, or the method must be declared to throw
//  this exception.
// -----------------------------------------------------------------------------------


class MatrixAlgebraException extends Exception {
    public MatrixAlgebraException () {super();}
    public MatrixAlgebraException (String s) {super(s);}
}


