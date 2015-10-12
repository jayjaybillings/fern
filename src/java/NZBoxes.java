// ------------------------------------------------------------------------------------------------------
//  Class NZBoxes to store coordinates & color of one isotope square
//  Entire diagram stored as a vector of these objects.  This is not
//  presently used, but may be needed as part of serialization of
//  a calculation configuration.
// ------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Vector;
import java.util.Properties;


class NZBoxes implements Serializable {
    public int x,y,width,height;
    public Color color;


    // ---------------------------------------------------------------------
    //  Constructor
    // ---------------------------------------------------------------------

    public NZBoxes(int x, int y, int width, int height, Color c) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.color = c;
    }

}   /* End class NZBoxes */
