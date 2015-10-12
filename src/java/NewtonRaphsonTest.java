// -----------------------------------------------------------------------------------------------------------------------------
//  Class to implement test of Newton-Raphson algorithm applied to partial equilibrium
//  for a single timestep.
//
//  Mike Guidry (November, 2009)
//
// -----------------------------------------------------------------------------------------------------------------------------

import java.io.*; 
import Jama.*;                    // The JAMA classes are in the Jama subdirectory

class NewtonRaphsonTest {
    
    static double K = 0.096281;
    static double ytil12 = 3.543901e-4;

    // --------------------------------------------------------------------------------------------------
    //  Constructor
    // --------------------------------------------------------------------------------------------------
    
    NewtonRaphsonTest () {    
  
    }
    
    
    // --------------------------------------------------------------------------------------------
    //  Main program to test Newton-Raphson.
    // --------------------------------------------------------------------------------------------

    public static void main(String[] args) {

                                                           
        // Take as an initial x vector
        double [] array6 = {0.028907, 0.039843, 0.012139}; 
            
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
        int maxit = 6;
        double tol = 1e-6;
        double xdev = 100;
        double [] returnx;
            
        while (xdev > tol && nit < maxit) {
            nit ++;
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
        
        if(nit < maxit){
            System.out.println("----------------------------------------------------------------------");
            System.out.println("Convergence obtained after "+nit+" iterations: Solution vector x ="); 
            test.getxMatrix().print(15,12);
            System.out.println("xdev="+xdev+", which is less than tolerance "+tol);
            System.out.println("----------------------------------------------------------------------\n");  
            
            double [] x = test.getx();
            System.out.println("ratio="+StochasticElements.gg.decimalPlace(6,x[0]*x[1]/x[2])+" compared with "
                    +StochasticElements.gg.decimalPlace(6,K)+" for equilibrium");
            System.out.println();
        } else {
            System.out.println("----------------------------------------------------------------------");
            System.out.println("Unconverged after "+nit+" iterations"); 
            System.out.println("----------------------------------------------------------------------\n");  
        }
        
    }   // End of main
    
    // Method to compute F(x) for the non-linear test    
    static double [] computeF(double [] x){
        double [] f = new double[x.length];  
        
        f[0] = x[0] - (K/x[1]) *x[2];
        f[1] = x[1] -K*x[2]/x[0];
        f[2] = x[0] + 3*ytil12 + 4*x[1] + 5*x[2] - 0.25;       
        
        return f;
    }
    
    // Method to compute the Jacobian for the non-linear test
    static double [][] computeJ(double [] x){  
        double [][] J = new double[x.length][x.length];
        
        J[0][0] = 1;
        J[0][1] = K*x[2]/x[1]/x[1];
        J[0][2] = -K/x[1];

        J[1][0] = K*x[2]/x[0]/x[0];
        J[1][1] = 1;
        J[1][2] = -K/x[0];
        
        J[2][0] = 1;
        J[2][1] = 4;
        J[2][2] = 5;
                
        return J; 
    }
            
}


