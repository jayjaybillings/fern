// -------------------------------------------------------------------------------------------------------------
//  The class GraphicsPad implements a 2-d contour plot of abundances in
//  conjunction with the classes ContourPlotter and ContourDisplay.
// --------------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Vector;
import java.util.Properties;


class GraphicsPad extends Canvas {

    Color    yellow=new Color(255,204,0);
    Color    orange=new Color(255,153,0);
    Color    red=new Color(204,51,0);
    Color    purple=new Color(153,102,153);
    Color    blue=new Color(102,153,153);
    Color    green=new Color(153,204,153);
    Color    white=new Color(255,255,255);
    Color    black=new Color(0,0,0);
    Color    gray51=new Color(51,51,51);
    Color    gray102=new Color(102,102,102);
    Color    gray153=new Color(153,153,153);
    Color    gray204=new Color(204,204,204);
    Color    gray250=new Color(250,250,250);

    Color isotopeColor = gray250;

    boolean numberFlag = true;          // Whether to show numbers in boxes
    double numberCutoff = 0;              // Minimum number to show in boxes

    protected Vector nzPlane = new Vector(256,256);       // Store graphics


    // --------------------------------------------------------------------
    //  paint method
    // --------------------------------------------------------------------

    public void paint(Graphics g){

        // Define fonts and corresponding fontmetrics for use in graph

        Font titleFont = new Font("Arial",Font.BOLD,12);
        Font smallFont = new Font("Arial",Font.PLAIN,11);
        Font realSmallFont = new Font("Arial",Font.PLAIN,9);
        Font bigFont = new Font("Arial",Font.BOLD,14);
        FontMetrics titleFontFontMetrics = getFontMetrics(titleFont);
        FontMetrics smallFontFontMetrics = getFontMetrics(smallFont);
        FontMetrics realSmallFontFontMetrics = getFontMetrics(realSmallFont);
        FontMetrics bigFontFontMetrics = getFontMetrics(bigFont);

        int leftx = 60;
        int topy = 10;
        int gwidth = 500;
        int gheight = 500;
        byte pmax = StochasticElements.pmaxPlot;
        byte nmax = StochasticElements.nmaxPlot;
        double maxValue = StochasticElements.maxValue;

        // Figure out spacing for isotope boxes

        int isotopeWidth = gwidth/nmax;
        int isotopeHeight = gheight/pmax;

        /* Is the following color contour stuff supplanted by stuff in ContourFrame? */

        // Determine contour levels

        double contourFraction[] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
        int length = contourFraction.length;
        double contourRange[] = new double[length];
        for (int i=0; i<= length-1; i++) {
            contourRange[i] = contourFraction[i]*StochasticElements.maxValue;
            System.out.println("contour="+contourRange[i]);
        }


        // Set contour colors (presently blue to green to yellow to
        // orange to red for increasing values)

        Color contourColor[] = new Color[length];
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



        // Loop over isotope boxes to fill diagram

        int x=0;
        int y=0;

        for (int i=1; i<=pmax-1; i++) {
            for (int j=1; j<= nmax-1; j++) {

                // Determine the contour color for this box
                for (int k=0; k<=length-1; k++) {
                    if(StochasticElements.pop[i][j] <= contourRange[k]) {
                        isotopeColor = contourColor[k];
                        break;
                    }
                }

                g.setColor(isotopeColor);
                x = leftx + (j-1)*isotopeWidth+1;
                y = topy + gheight - i*isotopeHeight+1;
                g.fillRect(x,y,isotopeWidth,isotopeHeight);
                g.setColor(black);
                g.drawRect(x,y,isotopeWidth,isotopeHeight);

                //  Store the information for this box in vector nzPlane
                //  Add population numbers if numberFlag is true and
                //  Z > 2, N > 2

                g.setFont(realSmallFont);
                g.setColor(black);
                String tempS = String.valueOf(StochasticElements.pop[i][j]);
                if (numberFlag  && i>2 && j>2 && StochasticElements.pop[i][j]>numberCutoff) {
                    g.drawString(String.valueOf(StochasticElements.pop[i][j]),
                       x + isotopeWidth/2
                       - realSmallFontFontMetrics.stringWidth(tempS)/2,
                       y + isotopeHeight/2
                       + realSmallFontFontMetrics.getHeight()/2 -2);
                }
                g.setFont(smallFont);

                // Numbers for bottom axis

                tempS = String.valueOf(j);
                if (i==1) {
                    g.drawString(tempS,x+isotopeWidth/2
                        - smallFontFontMetrics.stringWidth(tempS)/2,y+40);
                }
            }

            // Numbers for left axis

            String tempS = String.valueOf(i);
            g.drawString(String.valueOf(i),leftx-10
                - smallFontFontMetrics.stringWidth(tempS),
                y+isotopeHeight/2
                + smallFontFontMetrics.getHeight()/2 -3);
        }

         // Add axis labels

         g.setFont(bigFont);
         g.drawString("P",10,gheight/2
            +topy+bigFontFontMetrics.getHeight()-3);
         g.drawString("N",leftx+gwidth/2
            -bigFontFontMetrics.stringWidth("N")-3, gheight+topy+50);

         // Create contour legend

         int leftSide1 = leftx+gwidth+10;
         int leftSide2 = leftSide1 + 70;
         int topSide = topy+50+20;
         int boxSize = 10;
         int vshift = smallFontFontMetrics.getHeight()-boxSize/2 -1;

         g.setColor(gray51);
         g.setFont(titleFont);
         g.drawString("Contours:", leftSide1, topy+50);
         g.setFont(smallFont);
         String temp1 = "0";
         String temp2 = String.valueOf((int)contourRange[0]);
         String temp =  temp1 + " - " + temp2;
         g.drawString(temp,leftSide1,topSide);

         g.setColor(contourColor[0]);
         g.fillRect(leftSide2,topSide-vshift,boxSize,boxSize);
         g.setColor(gray51);
         g.drawRect(leftSide2,topSide-vshift,boxSize,boxSize);

         int vspacing = 15;
         for (int i=0; i<= length-2; i++) {
             temp1 = String.valueOf((int)contourRange[i]);
             temp2 = String.valueOf((int)contourRange[i+1]);
             temp = temp1 + " - " + temp2;
             g.drawString(temp,leftSide1,topSide+(i+1)*vspacing);
             g.setColor(contourColor[i+1]);
             g.fillRect(leftSide2,topSide+(i+1)*vspacing - vshift,boxSize,boxSize);
             g.setColor(gray51);
             g.drawRect(leftSide2,topSide+(i+1)*vspacing - vshift,boxSize,boxSize);
         }
    }

}   /* End class GraphicsPad */

