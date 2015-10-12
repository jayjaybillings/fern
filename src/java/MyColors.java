// ------------------------------------------------------------------------------------------------------
// Class to define some standard colors that can be accessed as class
// variables (MyColors.color) without need of object instantiation.
// ------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

class MyColors {

    // Define some colors
    
    public static Color AIyellow=new Color (255,204,0);
    public static Color AIorange=new Color(255,153,0);
    public static Color AIred=new Color(204,51,0);
    public static Color AIpurple=new Color(153,102,153);
    public static Color AIblue=new Color(102,153,153);
    public static Color AIgreen=new Color(153,204,153);
    public static Color gray51=new Color(51,51,51);
    public static Color gray102=new Color(102,102,102);
    public static Color gray153=new Color(153,153,153);
    public static Color gray190=new Color(190,190,190);
    public static Color gray204=new Color(204,204,204);
    public static Color gray220=new Color(220,220,220);
    public static Color gray235=new Color(235,235,235);
    public static Color gray240=new Color(240,240,240);
    public static Color gray245=new Color(245,245,245);
    public static Color gray250=new Color(252,252,252);
    public static Color yellowGray = new Color(255,255,250);
    public static Color blueGray = new Color(245,245,255);

    public static Color warnColorBG = AIyellow;
    public static Color dialogColor = yellowGray;
    public static Color helpColor = yellowGray;
    
    
    // Cardall color table
    
    public static double cx[] = {0.0, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.0};
    public static int cR[] = {0, 32, 128, 128, 224, 160, 255, 160, 255};
    public static int cG[] = {0, 32, 128, 32, 128, 64, 160, 160, 255};
    public static int cB[] = {0, 128, 224, 128, 224, 64, 160, 64, 160};
    
    // Guidry color table
    
    public static double gx[] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    public static int gR[] = {19, 35, 60, 93, 134, 177, 217, 245, 255, 245, 217};
    public static int gG[] = {93, 134, 177, 217, 245, 255, 245, 217, 177, 134, 93};
    public static int gB[] = {163, 228, 255, 228, 163, 93, 43, 15, 4, 1, 0};
    
    // Modified Guidry color table
    
    public static double ggx[] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    public static int ggR[] = {19, 35, 60, 93, 134, 177, 217, 245, 255, 245, 217};
    public static int ggG[] = {93, 134, 177, 217, 245, 255, 245, 217, 177, 134, 93};
    public static int ggB[] = {200, 228, 255, 228, 163, 93, 43, 15, 4, 1, 0};
    
    // Visit bluehot color table
    
    public static double bhx[] = {0.0, 0.3333, 0.6667, 1.0};
    public static int bhR[] = {0, 0, 0, 255};
    public static int bhG[] = {0, 0, 127, 255};
    public static int bhB[] = {0, 127, 255, 255};
    
    // Visit caleblack color table
    
    public static double cbx[] = {0.0, 0.1667, 0.3333, 0.5000, 0.6667, 0.8333, 1.0};
    public static int cbR[] = {0, 0, 0, 0, 255, 255, 255};
    public static int cbG[] = {0, 0, 255, 255, 255, 0, 0};
    public static int cbB[] = {0, 255, 255, 0, 0, 0, 255};
    
    // Visit calewhite color table
    
    public static double cwx[] = {0.0, 0.1667, 0.3333, 0.5000, 0.6667, 0.8333, 1.0};
    public static int cwR[] = {255, 0, 0, 0, 255, 255, 255};
    public static int cwG[] = {255, 0, 255, 255, 255, 0, 0};
    public static int cwB[] = {255, 255, 255, 0, 0, 0, 255};
    
    // Visit hot color table
    
    public static double hx[] = {0.0, 0.25, 0.50, 0.75, 1.0};
    public static int hR[] = {0, 0, 0, 255, 255};
    public static int hG[] = {0, 255, 255, 255, 0};
    public static int hB[] = {255, 255, 0, 0, 0};
    
    // Greyscale color table
    
    public static double greyx[] = {0.0, 1.0};
    public static int greyR[] = {0, 255};
    public static int greyG[] = {0, 255};
    public static int greyB[] = {0, 255};
    
    
    public MyColors(){
    
    }
    
    
    
    // -------------------------------------------------------------------------------------------------------
    //  Method returnRGB to return an RGB color for contour plotter as
    //  a function of the fraction x (lying between 0 and 1) of the max
    //  contour level.  To draw, e.g., a continuous horizontal
    //  color scale, call this method from within a loop that draws on a
    //  graphics object a series of short vertical lines, with the x coordinate
    //  increasing by 1 pixel each iteration, passing the argument x divided
    //  by the interval length (to normalize to unit x interval) to returnRGB
    //  in order to set the drawing color before each line draw. Which map
    //  is used is determined by the String variable popColorMap. The 
    //  boolean variable invertMap inverts the color map if it is true.
    // ----------------------------------------------------------------------------------------------------------

    public Color returnRGB (boolean invertMap, String whichColorMap, double x) {

        if (whichColorMap.compareTo("cardall") == 0){
            InterpolateColorTable ict = new InterpolateColorTable(invertMap,
                MyColors.cx, MyColors.cR, MyColors.cG, MyColors.cB);
            return ict.rgb(x);
        } else if (whichColorMap.compareTo("guidry") == 0) {
            InterpolateColorTable ict = new InterpolateColorTable(invertMap,
                MyColors.gx, MyColors.gR, MyColors.gG, MyColors.gB);
            return ict.rgb(x); 
        } else if (whichColorMap.compareTo("guidry2") == 0) {
            InterpolateColorTable ict = new InterpolateColorTable(invertMap,
                MyColors.ggx, MyColors.ggR, MyColors.ggG, MyColors.ggB);
            return ict.rgb(x); 
        } else if (whichColorMap.compareTo("bluehot") == 0) {
            InterpolateColorTable ict = new InterpolateColorTable(invertMap,
                MyColors.bhx, MyColors.bhR, MyColors.bhG, MyColors.bhB);
            return ict.rgb(x); 
        } else if (whichColorMap.compareTo("caleblack") == 0) {
            InterpolateColorTable ict = new InterpolateColorTable(invertMap,
                MyColors.cbx, MyColors.cbR, MyColors.cbG, MyColors.cbB);
            return ict.rgb(x); 
        } else if (whichColorMap.compareTo("calewhite") == 0) {
            InterpolateColorTable ict = new InterpolateColorTable(invertMap,
                MyColors.cwx, MyColors.cwR, MyColors.cwG, MyColors.cwB);
            return ict.rgb(x); 
        } else if (whichColorMap.compareTo("hot") == 0) {
            InterpolateColorTable ict = new InterpolateColorTable(invertMap,
                MyColors.hx, MyColors.hR, MyColors.hG, MyColors.hB);
            return ict.rgb(x); 
        } else if (whichColorMap.compareTo("greyscale") == 0) {
            InterpolateColorTable ict = new InterpolateColorTable(invertMap,
                    MyColors.greyx, MyColors.greyR, MyColors.greyG, MyColors.greyB);
            return ict.rgb(x); 
        } else {
            double x0R = 0.8;
            double x0G = 0.5;
            double x0B = 0.2;
            double aR = 0.5;
            double aG = 0.5;
            double aB = 0.3;
    
            int red = (int) (255*Math.exp( -(x-x0R)*(x-x0R)/aR/aR ));
            int green = (int) (255*Math.exp( -(x-x0G)*(x-x0G)/aG/aG ));
            int blue = (int) (255*Math.exp( -(x-x0B)*(x-x0B)/aB/aB ));
            return new Color(red,green,blue);
        }

    }




}  /* End class MyColors */

