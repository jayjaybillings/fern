// ----------------------------------------------------------------------------------------------------------------
//  Class to accept a string and to typeset it horizontally in graphics mode
//  with arbitrary constant spacing between letters.  Variables passed
//  to contructor:
//        x = x coordinate for center of horizontal string
//        y = y coordinate for horizontal string base
//        leading = desired spacing between horizontal letters
//        f = font to typeset letters in
//        fm = FontMetrics associated with the font f
//        g = graphics context
//  To use, instantiate with appropriate arguments.  For example,
//
//     HorString hs = new HorString("Hi",10,20,2,myFont,myFontMetrics,g);
// -----------------------------------------------------------------------------------------------------------------


import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

class HorString {

    int w, length, newx;
    String letter;

    // ----------------------------------------------------------------------
    //  Constructor
    // ----------------------------------------------------------------------

    HorString(String s, int x, int y, int leading, Font f, FontMetrics fm, Graphics g) {

        length = s.length();
        w = 0;
        for (int i=0; i<length; i++) {
            letter = s.substring(i,i+1);
            w += fm.stringWidth(letter);
        }
        w += (length-1)*leading;
        g.setFont(f);
        newx = x-w/2;
        for (int i=0; i<length; i++) {
            letter = s.substring(i,i+1);
            g.drawString(letter, newx, y);
            newx += (fm.stringWidth(letter) + leading);
        }
    }

}   /*  End class HorString  */

