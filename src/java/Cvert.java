// ----------------------------------------------------------------------------------------------------------------
//  Class to implement various type casts and conversions. All method are
//  static so they can be invoked directly from the class:  Cvert.method()
//  without having to instantiate (but they can be instantiated also).
// -----------------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.Vector;
import java.util.Properties;
import java.text.*;


class Cvert {

    static final double LOG10 = 0.434294482;                //  Conversion natural log to log10
    static final double MEV = 931.494;                          // Conversion of amu to MeV
    static final double ECON = 9.5768e17;                    // Convert MeV/nucleon/s to erg/g/s
    static final double ERGTOMEV = 1.60217733e-6;   // ergs per MeV

    // Access decimal truncator by Cvert.gg.decimalPlace(places, number)
    
    static GraphicsGoodies2 gg = new GraphicsGoodies2();
    
    
    public void Cvert(){ }
    
    
    // ----------------------------------------------------------------------
    //  Method to call system exit
    // ----------------------------------------------------------------------

    public static void callExit(String message) {
        System.out.println();
        System.out.println(message);
        System.out.println();
        System.exit(1);
    }
    
    
    // --------------------------------------------------------------------------------------
    //  Method returnSymbol to return element symbol given the
    //  proton number.
    // --------------------------------------------------------------------------------------

    static String returnSymbol (int z) {

        String [] symbolString = {"n","H","He","Li","Be","B","C","N",
                "O","Fl","Ne","Na","Mg","Al","Si","P","S","Cl",
                "Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co",
                "Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb",
                "Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag",
                "Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La",
                "Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho",
                "Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir",
                "Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr",
                "Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk",
                "Cf","Es","Fm","Md","No","Lr"};

        return symbolString [z];
    }
    
    
    // -----------------------------------------------------------------------------------------------
    //  Static method stringToDouble to convert a string to a double.
    //  To go the other way use String.valueOf(double d)
    // -----------------------------------------------------------------------------------------------

    static double stringToDouble (String s) {
        Double mydouble=Double.valueOf(s);    // String to Double (object)
        return mydouble.doubleValue();              // Return primitive double
    }


    // -----------------------------------------------------------------------------------
    //  Static method stringToInt to convert a string to an int.
    //  To go the other way, use String.valueOf(int i)
    // -----------------------------------------------------------------------------------

    static int stringToInt (String s) {
        Integer myInt=Integer.valueOf(s);     // String to Integer (object)
        return myInt.intValue();                    // Return primitive int
    }
    
    
    // --------------------------------------------------------------------------------------------------
    //  Static method to convert a string to a Boolean. Arguments should
    //  be "true" or "false" without case sensitivity.  To go the other way
    //  use String.valueOf(boolean b).
    // --------------------------------------------------------------------------------------------------
    
    static Boolean stringToBoolean(String s){
        return Boolean.valueOf(s);
    }
    
    
         
    // -----------------------------------------------------------------------------------------------------------------------
    //  Utility method to replace whitespace in string with user-specified character given
    //  by the argument replace.
    // ------------------------------------------------------------------------------------------------------------------------

   static String replaceWhiteSpace(String inputString, String replace){
    
        int count = 0;
        String temp = "";
        inputString = inputString.trim();    // Trim leading and trailing whitespace
        
        StringTokenizer tk = new StringTokenizer(inputString);
        while(tk.hasMoreTokens()){
            if(count>0) temp += replace;
            temp +=  tk.nextToken();  
            count ++;
        }
            
        return temp;
    }
    
    
    // -----------------------------------------------------------------------------------------------------------------------
    //  Utility method to replace all instances of one character in a string with a 
    //  second character. It also trims leading and trailing whitespace.
    // ------------------------------------------------------------------------------------------------------------------------

   static String replaceThisWithThat(String inputString, String target, String replace){
        
        int count = 0;
        String temp = "";
        inputString = inputString.trim();    // Trim leading and trailing whitespace
        
        StringTokenizer tk = new StringTokenizer(inputString, target);
        while(tk.hasMoreTokens()){
            if(count>0) temp += replace;
            temp +=  tk.nextToken();  
            count ++;
        }
            
        return temp;
    }
    
    
    // -----------------------------------------------------------------------------------
    //  Static method to return the base-10 log of a double.  
    //  Overloaded (see methods below for int and long int)
    // -----------------------------------------------------------------------------------   
    
    static double log10(double number) {
        return LOG10*Math.log(number);
     }
     
     
    // -----------------------------------------------------------------------------------
    //  Static method to return the base-10 log of an int
    // -----------------------------------------------------------------------------------   
    
    static double log10(int number) {
        return LOG10*Math.log((double)number);
     }
     
     
    // -----------------------------------------------------------------------------------
    //  Static method to return the base-10 log of a long int
    // -----------------------------------------------------------------------------------   
    
    static double log10(long number) {
        return LOG10*Math.log((double)number);
     }
     
     
    // -----------------------------------------------------------------------------------
    //  Static method to convert ergs to MeV
    // -----------------------------------------------------------------------------------  
     static double ergToMeV(double erg){
        return erg/ERGTOMEV;
     }
     
     
    // -----------------------------------------------------------------------------------
    //  Static method to convert MeV to ergs
    // -----------------------------------------------------------------------------------  
     static double MeVToErg(double mev){
        return mev*ERGTOMEV;
     }
     

}