// -----------------------------------------------------------------------------------------------------------
//  Class to accept a string and to typeset it vertically in graphics mode.
//  Variables passed to contructor:
//        x = x coordinate for left side of vertical string
//        y = y coordinate for center of vertical string
//        leading = extra spacing between vertical letters (can be negative)
//        f = font to typeset letters in
//        fm = FontMetrics associated with the font f
//        g = graphics context
//  To use, instantiate with appropriate arguments.  For example,
//
//     VertString vs = new VertString("Hi",10,20,2,myFont,myFontMetrics,g);
// -------------------------------------------------------------------------------------------------------------


import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;


class VertString {

    int h, voffset, length, newy, newx, xshift;
    String letter;

    // --------------------------------------------------------------------------
    //  Constructor
    // --------------------------------------------------------------------------

    VertString(String s, int x, int y, int leading, Font f, FontMetrics fm, Graphics g) {

        h = fm.getHeight();
        length = s.length();
        voffset = y - (int)(0.8*(h + 0.5) + leading)*length/2;
        g.setFont(f);

        for (int i=0; i<length; i++) {
            letter = s.substring(i,i+1);
            newy = voffset + i*(int)(0.8*h + leading);
            xshift = fm.stringWidth(letter)/2;
            newx = x-xshift;
            g.drawString(letter, newx, newy);
        }
    }

}   /*  End class VertString  */

