// ------------------------------------------------------------------------------------------------------
// Class to define population vector components
// ------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

class IsoVector {
    
     int Z;
     int N;
     double A;
     String symbol;
     boolean isEquil;
        
    public IsoVector(int Z, int N, String symbol){     
        this.Z = Z;
        this.N = N;
        this.A = (double)(Z+N);
        this.symbol = symbol;
        this.isEquil = false;
    }

}  /* End class IsoVector */

