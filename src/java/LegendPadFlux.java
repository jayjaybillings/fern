// ----------------------------------------------------------------------------------------------------------------------------
//  The class LegendPadFlux implements a legend for 2-d contour plot of fluxes.
// ----------------------------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Properties;


class LegendPadFlux extends Canvas {

    // Define fonts and corresponding fontmetrics for use in graph

    Font titleFont = new Font("SansSerif",Font.BOLD,11);
    Font smallFont = new Font("SansSerif",Font.PLAIN,11);
    Font realSmallFont = new Font("SansSerif",Font.PLAIN,9);
    Font bigFont = new Font("SansSerif",Font.BOLD,14);
    FontMetrics titleFontFontMetrics = getFontMetrics(titleFont);
    FontMetrics smallFontFontMetrics = getFontMetrics(smallFont);
    FontMetrics realSmallFontFontMetrics = getFontMetrics(realSmallFont);
    FontMetrics bigFontFontMetrics = getFontMetrics(bigFont);

    double [] contourRange;
    Color [] contourColor;
    int length;

    String contourLegend;


    // -------------------------------------------------------------------------
    //  Constructor
    // -------------------------------------------------------------------------

    public LegendPadFlux(String contourLegend, double [] contourRange, Color [] contourColor) {

        super();
        setSize(95,190);
        setVisible(true);

        this.contourLegend = contourLegend;
        this.contourRange = contourRange;
        this.contourColor = contourColor;

        length = contourRange.length;
        this.setBackground(MyColors.gray220);
        repaint();
    }


    // --------------------------------------------------------------------
    //  paint method
    // --------------------------------------------------------------------

    public void paint(Graphics g){

        int topy = 10;
        int vspacing = 15;

        // Create contour legend
        
        MyColors mc = new MyColors();

        for(int i=0; i<158; i++) {
            g.setColor( mc.returnRGB(StochasticElements.fluxColorInvert, 
                StochasticElements.fluxColorMap, (double)(158-i)*0.00632911 ) );
            //g.setColor( ShowIsotopeFlux.cd.returnRGB( (double)(158-i)*0.00633 ) );
            g.drawLine(73, i+27, 85, i+27);
        }

        int leftSide1 = 5;
        int leftSide2 = leftSide1 + 60;
        int topSide = 35;
        int boxSize = 10;
        int vshift = smallFontFontMetrics.getHeight()-boxSize/2 -1;

        String XYlegend = contourLegend;

        g.setColor(Color.white);
        g.setFont(titleFont);
        g.drawString(XYlegend, leftSide1, topy+5);
        g.setColor(Color.white);
        g.setFont(smallFont);

        String temp = Cvert.gg.decimalPlace(3,contourRange[0]);
        g.drawString(temp,leftSide1,topSide+(length-1)*vspacing);

        for (int i=0; i<length-1; i++) {
            temp =
                Cvert.gg.decimalPlace(3,contourRange[length-1-i]);

            g.drawString(temp,leftSide1,topSide+(i)*vspacing);
        }

    }

}   /* End class LegendPadFlux */

