
/*
Class to implement interpolation in RGB color table specified as a sequence of RGB values.
*/

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;


public class InterpolateColorTable {

    private SplineInterpolator si;
    private double x[]; 
    private double R[], G[], B[];
    private int red, green, blue;
    private String errString;
    private int lenx;
    private boolean invertMap;


    // --------------------------------------------------------------------------------------------------------------------------------------    
    //  Public constructor. The array x is an array of interpolant values and the arrays R, G, and B 
    //  are arrays of the same length as x containing the corresponding red, green, and blue color 
    //  values (0-255). The boolean invertMap inverts the color mapping if it is true.
    // --------------------------------------------------------------------------------------------------------------------------------------    
     
    public InterpolateColorTable (boolean invertMap, double x[], int R[], int G[], int B[]) {
    
        // Check for obvious inconsistencies
        
        lenx = x.length;
        int lenR = R.length;
        int lenG = G.length;
        int lenB = B.length;
        
        errString = "** InterpolateColorTable error: incompatible array lengths x, R, G, B.";
        
        if(lenx != lenR || lenx != lenG || lenx != lenB) Cvert.callExit(errString);
        
        // Copy arguments to class variables for use in rgb method
        
        this.invertMap = invertMap;
        this.x = x; 
        this.R = new double [lenx];
        this.G = new double [lenx];
        this.B = new double [lenx];
        
        // Must cast integer arrays to doubles because spline interpolator expects doubles
        
        for(int i=0; i<lenx; i++){
            if(invertMap){
                this.R[i] =  (double) R[lenx-i-1];
                this.G[i] = (double) G[lenx-i-1];
                this.B[i] =  (double) B[lenx-i-1];             
            } else {
                this.R[i] =  (double) R[i];
                this.G[i] = (double) G[i];
                this.B[i] =  (double) B[i]; 
            }
        }
        
    }
    
    
    // --------------------------------------------------------------------------------------------------------------------------------------    
    // Method to interpolate and return a Color object specifying the RGB color.
    // --------------------------------------------------------------------------------------------------------------------------------------
    
    public Color rgb (double xval) {
    
        // Check for obvious inconsistencies
        
        errString = "** InterpolateColorTable.rgb(arg): arg=";
        errString += xval+" out of table bounds (" +x[0] +"-" + x[lenx-1]+")"; 
        
        if(xval < x[0] || xval > x[lenx-1]) Cvert.callExit(errString);
    
        // Create an interpolation object
        
        si = new SplineInterpolator();
        
        // Interpolate red value (cast to int because Math.round returns long)   
        
        si.spline(x, R);
        red = (int) Math.round(si.splint(xval));
        
        // Interpolate green value      
        
        si.spline(x, G);
        green = (int) Math.round(si.splint(xval));
        
        // Interpolate blue value      
        
        si.spline(x, B);
        blue = (int) Math.round(si.splint(xval));
        
        // Catch and process exception if interpolated colors are out of bounds 0-255
        Color tryColor;
        try {
            tryColor = new Color(red, green, blue);
        } catch (IllegalArgumentException e) {
            System.out.println(e);
            System.out.println("  Class InterpolateColorTable: Illegal color, R="
                    +red+" B="+blue+" G="+green +".  Set to Color.WHITE and continue." );
            tryColor = Color.WHITE;
        }
        
        //return new Color(red, green, blue);
        
        return tryColor;
        
    }
    
    
    // ----------------------------------------------------------------------------------------------------------------------------------------    
    //  Main method to test rgb method. Execute with "java InterpolateColorTable arg" where arg
    //  is a value in the range of the color table.
    // ----------------------------------------------------------------------------------------------------------------------------------------
    
    public static void main(String[] args) {

        if (args.length != 1) {      // Check for fileName argument
            System.err.println();
            System.err.println("Usage: java InterpolateColorTable <xvalue 0-1>");
            System.err.println();
            System.exit(1);
        }
        
        // Set up a color table
        
        double x[] = {0.0, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.0};
        int R[] = {0, 32, 128, 128, 224, 160, 255, 160, 255};
        int G[] = {0, 32, 128, 32, 128, 64, 160, 160, 255};
        int B[] = {0, 128, 224, 128, 224, 64, 160, 64, 160};
        
        // Create an object to interpolate in it
      
        InterpolateColorTable test = new InterpolateColorTable(false, x, R, G, B);
        
        Color c = test.rgb(Cvert.stringToDouble(args[0]));
        
        System.out.println("red="+c.getRed()+" green="+c.getGreen()+" blue="+c.getBlue());
        
    }

}