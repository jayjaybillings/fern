// -------------------------------------------------------------------------
//  Class LegendTester
// -------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

public class LegendTester extends Frame {


    public static void main (String[] args) {
        LegendTester lt = new LegendTester();
    }


    public LegendTester() {

        // Determine contour levels

        this.setSize(100,200);

        int numberContours = 10;
        double [] contourFraction = new double[numberContours];
        double [] contourRange = new double[numberContours];
        Color [] contourColor = new Color[numberContours];

        for (int i=0; i<numberContours; i++) {
          contourFraction[i] = 0.1*(i+1);
          contourRange[i] = contourFraction[i]*1000;
          System.out.println("ContourFrame: i="+i
                    +"contourRange="+contourRange[i]);
        }

        // Set contour colors (presently gray to yellow to
        // blue to red for increasing values)

        contourColor[9] = new Color(255,0,0);
        contourColor[8] = new Color(255,150,150);
        contourColor[7] = new Color(255,200,200);
        contourColor[6] = new Color(0,0,255);
        contourColor[5] = new Color(150,150,255);
        contourColor[4] = new Color(220,220,255);
        contourColor[3] = new Color(255,255,0);    // yellow
        contourColor[2] = new Color(255,255,180);  // yellow
        contourColor[1] = new Color(255,255,230);  // yellow
        contourColor[0] = new Color(245,245,245);  // graywhite

        LegendPad lp = new LegendPad(contourRange,contourColor);

        this.add(lp, "Center");
        this.show();
    }

}

