// -----------------------------------------------------------------------------------------------------------------------------
//  Class to implement Newton-Raphson iteration to solve non-linear sets of equations.
//  Requires the JAMA package (see http://math.nist.gov/javanumerics/jama/), which
//  implements a basic linear algebra package for java.  A main program is included
//  as a test driver (execute with java NewtonRaphson), but the usual usage in another
//  program would be to instantiate a NewtonRaphson object like
//
//          NewtonRaphson nr = new NewtonRaphson(J, F, x);
//
//  and then access its methods.  The main program gives a full example of solving
//  a simple non-linear set of equations by successive iteration.
//
//  Mike Guidry (August, 2009)
//
// -----------------------------------------------------------------------------------------------------------------------------

import java.io.*; 
import Jama.*;                    // The JAMA classes are in the Jama subdirectory

class NewtonRaphson {
    
    Matrix J;
    Matrix delx;
    Matrix F;
    Matrix x;
    int Jrow;
    int Jcol;
    
    // --------------------------------------------------------------------------------------------------
    //  Use constructor to set up to solve matrix equation J.delx = - F
    //  for the unknown vector delx, given the matrix J and the vector F.
    // --------------------------------------------------------------------------------------------------
    
    NewtonRaphson (double[][] JJ, double [] FF, double [] xx) {
        
        // Convert Jrow x Jcol array J to Matrix object
        J = new Matrix(JJ);
        Jrow = J.getRowDimension();
        Jcol = J.getColumnDimension();
        
        // Convert Jrow-dimensional column vector FF to Jrow x 1 Matrix object F
        double temp[][];
        int Flen = 0; 
        // Check for conformability of matrix and column vector
        try{ 
            Flen = FF.length;
            if(Flen != Jrow){
                String es = "\nColumns in J ("+Jcol+") differ from rows in F ("+Flen+"). End execution.\n";
                throw new conformabilityException(es);
            }
        } 
        catch(conformabilityException e){
            System.out.println("\nconformabilityException in Newton-Raphson: "+e.getMessage());
            System.exit(0);
        }    
        temp = new double[Flen][];
        for(int i=0; i<Flen; i++){
            temp[i] = new double[1];
            temp[i][0] = FF[i];
        }
        F = new Matrix(temp);
        
        // Convert Jrow-dimensional column vector xx to Jrow x 1 Matrix object x

        double temp2[][];
        try{ 
            Flen = xx.length;
            if(Flen != Jcol){
                String es = "\nColumns in J ("+Jcol+") differ from rows in x ("+Flen+"). End execution.\n";
                throw new conformabilityException(es);
            }
        } 
        catch(conformabilityException e){
            System.out.println("\nconformabilityException in Newton-Raphson: "+e.getMessage());
            System.exit(0);
        }
        temp2 = new double[Flen][];
        for(int i=0; i<Flen; i++){
            temp2[i] = new double[1];
            temp2[i][0] = xx[i];
        }
        x = new Matrix(temp2);
        
        // Initialize the delx matrix.  It will only be filled when the doIteration method is invoked.
        delx = new Matrix(Flen, Flen);
    }
    
    
    // ----------------------------------------------------------------------------------------------------
    //  Method to do one Newton-Raphson iteration (solve for delx in
    //  the matrix equation J.delx = - F, given the current entries in the
    //  the matrix J and column vector F). Returns the updated solution
    //  vector x+delx as a 1d array.
    // ----------------------------------------------------------------------------------------------------
    
    public double [] doIteration(){        
        delx = J.solve(F.times(-1));         // Matrix solve for delx  
        x.plusEquals(delx);                    // Replace x -> x + delx in place
        return x.getRowPackedCopy();  // Return updated solution vector
    }
    
    
    // ---------------------------------------------------------------------------------------------------
    // Method to return the current J matrix as a Matrix object
    // ---------------------------------------------------------------------------------------------------
    
    public Matrix getJMatrix(){
        return J;
    }
    
    
    // ---------------------------------------------------------------------------------------------------
    // Method to return the current values of the J matrix as a 2d array
    // ---------------------------------------------------------------------------------------------------
    
    public double [][] getJ(){
        return J.getArray();
    }
    
    
    // ---------------------------------------------------------------------------------------------------------
    // Method to return the current values of the delx vector as a Matrix object
    // ---------------------------------------------------------------------------------------------------------
    
    public Matrix getdelxMatrix(){
        return delx;
    }
    
    
    // ---------------------------------------------------------------------------------------------------
    // Method to return the current values of the delx vector as a 1d array
    // ---------------------------------------------------------------------------------------------------
    
    public double [] getdelx(){
        return delx.getRowPackedCopy();
    }
        
        
    // ------------------------------------------------------------------------------------------------------
    // Method to return the current values of the F vector as a Matrix object
    // ------------------------------------------------------------------------------------------------------
    
    public Matrix getFMatrix(){
        return F;
    }
    
        
    // -------------------------------------------------------------------------------------------------
    // Method to return the current values of the F vector as a 1d array
    // -------------------------------------------------------------------------------------------------
    
    public double [] getF(){
        return F.getRowPackedCopy();
    }
    
    
    // ------------------------------------------------------------------------------------------------------
    // Method to return the current values of the x vector as a Matrix object
    // ------------------------------------------------------------------------------------------------------
    
    public Matrix getxMatrix(){
        return x;
    }
    
        
    // -------------------------------------------------------------------------------------------------
    // Method to return the current values of the x vector as a 1d array
    // -------------------------------------------------------------------------------------------------
    
    public double [] getx(){
        return x.getRowPackedCopy();
    }
    
    
    // -------------------------------------------------------------------------------------------------
    //  Method to update the Matrix J with an array of new values
    // -------------------------------------------------------------------------------------------------
    
    public void updateJMatrix(double [][] newJ){
        for(int i=0; i<Jrow; i++){
            for(int j=0; j<Jcol; j++){
                J.set(i, j, newJ[i][j]);
            }     
        }       
    }
    
    
    // -------------------------------------------------------------------------------------------------
    //  Method to update the F vector with an array of new values
    // -------------------------------------------------------------------------------------------------
    
    public void updateFMatrix(double [] newF){
        for(int i=0; i<Jcol; i++){
            F.set(i, 0, newF[i]);    
        }       
    }
    
    
    // -------------------------------------------------------------------------------------------------
    //  Method to return a measure of how close we are to the solution.
    //  Returns the rms deviation of the components of delx from zero.
    // -------------------------------------------------------------------------------------------------
    
    public double getxdev() {
        double xdev = 0;
        for(int i=0; i<Jcol; i++){
            double dx = delx.get(i,0);
            xdev += dx*dx;
        } 
        return Math.sqrt(xdev);
    }
        
    
    // --------------------------------------------------------------------------------------------
    //  Main program to test (java NewtonRaphson invokes this).
    // --------------------------------------------------------------------------------------------

    public static void main(String[] args) {
        
        // Test Newton-Raphson constructor
        
        double[][] j = {{1,2,3},{4,5,6},{7,8,10}};  // Initial Jacobian matrix
        double [] f = {2,2,3};                           // Initial F vector
        double [] x = {3, 5, 8};                        // Initial x vector
            
        NewtonRaphson nr = new NewtonRaphson(j, f, x);
        
        System.out.println("\nJ matrix:");
        nr.getJMatrix().print(10,6);
        System.out.println("F vector");
        nr.getFMatrix().print(10,6);
        System.out.println("x vector");
        nr.getxMatrix().print(10,6);
        
        // Test one Newton-Raphson iteration

        double [] delx = nr.doIteration();
            
        System.out.println("Solution delx after one iteration:");
        nr.getdelxMatrix().print(20,16);    
            
        System.out.println("\nCheck solution by matrix multiplication:\n\nJ.delx=");
        nr.getJMatrix().times(nr.getdelxMatrix()).print(20,16);
        System.out.println("Which should be -F:");
        nr.getFMatrix().times(-1).print(20,16);  
        System.out.println("updated x vector = x + delx");
        nr.getxMatrix().print(10,6);
        
                
        // After one Newton-Raphson iteration with doIteration(), delx holds the solution to J.delx = -F, 
        // x holds the updated vector x + delx, and J and F still have their original values. Before the 
        // next iteration step F(x) and J(x) must be computed with the new values in the updated 
        // vector x and passed to the NewtonRaphson instance using updateFMatrix() and updateJMatrix() 
        // methods. Then doIteration() can be invoked again, and so on, until the vector x converges to 
        // nearly the same value in successive iterations.
        
            
        // Test of Newton-Raphson iteration on a non-linear problem to convergence.  The problem is
        //  to solve the set of 3 equations
        //
        //       0.5 x_0^3 + 0.2 x_1 + x_2 = 0.5
        //       x_1 + 0.1 x_2 = 0.2
        //       0.3 x_0^2 +0.1 x_1 = -0.3
        //
        //  by the Newton-Raphson method for the unknown vector x = (x_0, x_1, x_2).  Thus the vector 
        //  F = (F_1, F_2, F_3)  has components F_1 = 0.5 x_0^3 + 0.2 x_1 + x_2 - 0.5, etc., and the
        //  Jacobian matrix is (partial F/ partial x)_{i j} = partial F_i / partial x_j and is given by
        //
        //                1.5 x_0^2    0.2     1    
        //         J =        0           1      0.1 
        //                 0.6 x_0      0.1     0  
        //
        //  The methods computeF(x) and computeJ(x) defined below calculate the F and J matrices for
        //  this case as a function of x.
        
        System.out.println("----------------------------------------------------------------------");
        System.out.println("\nExample: Solve the non-linear system of 3 equations\n");
        System.out.println("0.5 x_0^3 + 0.2 x_1 + x_2 = 0.5");
        System.out.println("x_1 + 0.1 x_2 = 0.2");
        System.out.println("0.3 x_0^2 +0.1 x_1 = -0.3");
        System.out.println("\nfor the unknown vector x=(x_0, x_1, x_2), given an initial guess x=(0,1,0).");
        System.out.println("Writing this set of equations as F(x)=0, the 3-component vector F(x) is\n");
        System.out.println("       F_0       0.5 x_0^3 + 0.2 x_1 + x_2 - 0.5");
        System.out.println(" F =   F_1   =   x_1 + 0.1 x_2 - 0.2");
        System.out.println("       F_2       0.3 x_0^2 + 0.1 x_1 + 0.3");
        System.out.println("\nand the 3x3 Jacobian matrix J(x)_{ij} = partial F_i / partial x_j is\n");
        System.out.println("      1.5 x_0^2    0.2      1");
        System.out.println(" J =     0          1      0.1" );
        System.out.println("      0.6 x_0      0.1      0");
        System.out.println("\nwhich isn't constant since the problem is non-linear. For x_init=(1,0,1)");
                                                           
        // Take as an initial x vector
        double [] array6 = {1,0,1}; 
            
        // The current value of F is returned by the method computeF and the current value of
        // J is returned by the method computeJ, which are defined below.  Thus
            
        NewtonRaphson test = new NewtonRaphson(computeJ(array6), computeF(array6), array6);
        
        System.out.println("the initial J, F, and x matrices computed from these expressions are:");
        test.getJMatrix().print(10,6);
        test.getFMatrix().print(10,6);
        test.getxMatrix().print(10,6);
        System.out.println("Solve by Newton-Raphson iteration until convergence tolerance satisfied\n");
            
        // Now iterate until the convergence measure NewtonRaphson.getxdev() is less than
        // a tolerance parameter.
            
        int nit = 0;
        double tol = 1e-6;
        double xdev = 100;
        double [] returnx;
            
        while (xdev > tol) {
            nit += 1;
            returnx = test.doIteration();  // returnx holds the updated vector x + delx
            xdev = test.getxdev();
            System.out.println("----- Iteration "+nit+" (xdev="+xdev+"):");
            System.out.println("\ndelx =");
            test.getdelxMatrix().print(15,12);
            System.out.println("xnew =");
            test.getxMatrix().print(15,12);
            System.out.println("Check: solution requires F(x) = 0. After this iteration F(x) =");
            test.getFMatrix().print(15,12);
            
            // Update the F and J matrices based on the new vector x + delx
            test.updateFMatrix(computeF(returnx));
            test.updateJMatrix(computeJ(returnx));
        }
        
        System.out.println("----------------------------------------------------------------------");
        System.out.println("Convergence obtained after "+nit+" iterations: x ="); 
        test.getxMatrix().print(15,12);
        System.out.println("xdev="+xdev+", which is less than tolerance "+tol);
        System.out.println("----------------------------------------------------------------------\n");
        
        
        // Example of solving a single equation x^3 - x^2 = 1.
        
        System.out.println("\nExample: solving a single equation in one unknown:\n");
        
        // Matrices all have a single entry in this case
        
        double [] xx = {1};      // Initial x vector
        double[][] jj = {{1}};     // Initial Jacobian matrix with xx=1
        double [] ff = {-1};       // Initial F vector with xx=1
        
        NewtonRaphson single = new NewtonRaphson(jj, ff, xx);
        
        System.out.println("the initial J, F, and x matrices (all with a single entry) are:");
        single.getJMatrix().print(10,6);
        single.getFMatrix().print(10,6);
        single.getxMatrix().print(10,6);
        System.out.println("Solve by Newton-Raphson iteration until convergence tolerance satisfied\n");
        
        nit = 0;
        tol = 1e-6;
        xdev = 100;
        double [] returnxSingle;
        
        while (xdev > tol) {
            nit += 1;
            returnxSingle = single.doIteration();  // returnxSingle holds the updated vector x + delx
            xdev = single.getxdev();
            System.out.println("----- Iteration "+nit+" (xdev="+xdev+"):");
            System.out.println("\ndelx =");
            single.getdelxMatrix().print(15,12);
            System.out.println("xnew =");
            single.getxMatrix().print(15,12);
            System.out.println("Check: solution requires F(x) = 0. After this iteration F(x) =");
            single.getFMatrix().print(15,12);
            
            // Update the F and J matrices based on the new vector x + delx
            
            ff[0] = returnxSingle[0]*returnxSingle[0]*returnxSingle[0] - returnxSingle[0]*returnxSingle[0] -1;
            jj[0][0] = 3*returnxSingle[0]*returnxSingle[0] - 2*returnxSingle[0];
            single.updateFMatrix(ff);
            single.updateJMatrix(jj);
        }   
            
        System.out.println("----------------------------------------------------------------------");
        System.out.println("Convergence obtained after "+nit+" iterations: x ="); 
        single.getxMatrix().print(15,12);
        System.out.println("xdev="+xdev+", which is less than tolerance "+tol);
        System.out.println("----------------------------------------------------------------------\n");
        
        
    } 
    
    // Method to compute F(x) for the non-linear test    
    static double [] computeF(double [] x){
        double [] f = new double[x.length];  
        f[0] = 0.5*x[0]*x[0]*x[0] + 0.2*x[1]  + x[2] - 0.5;
        f[1] = x[1] + 0.1*x[2] - 0.2;
        f[2] = 0.3*x[0]*x[0] + 0.1*x[1] - 0.3;     
        return f;
    }
    
    // Method to compute the Jacobian for the non-linear test
    static double [][] computeJ(double [] x){  
        double [][] J = new double[x.length][x.length];
        J[0][0] = 1.5*x[0]*x[0];
        J[0][1] = 0.2;
        J[0][2] = 1;
        J[1][0] = 0;
        J[1][1] = 1;
        J[1][2] = 0.1;
        J[2][0] = 0.6*x[0];
        J[2][1] = 0.1;
        J[2][2] = 0;    
        return J; 
    }
            
}


// ----------------------------------------------------------------------------------------------------------------------------------------
//  Custom-defined exception to be thrown if a matrix to be multiplied by a column
//  vector does not have have number of matrix columns equal to number of vector entries.
// ----------------------------------------------------------------------------------------------------------------------------------------

class conformabilityException extends Exception{
    
    public conformabilityException(){super();}
    public conformabilityException(String s){super(s);}
    
}
