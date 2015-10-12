// --------------------------------------------------------------------------------------------------------------------------
//  Class GraphicsGoodies2 contains various methods to help with
//  graphics operations in applets and programs.  To use methods of this class from
//  another class, be sure that the class file GraphicsGoodies2.class is in
//  the path (e.g., place in the same directory as the other class), instantiate
//  this class and then access its methods.  Example:  In another class,
//  to access the decimalPlace method of this class,
//
//      GraphicsGoodies2 gg = new GraphicsGoodies2(); // Instantiate this class
//      double num = 3.1416;
//      String string = gg.decimalPlace(2,num);       // Access its methods
//
//  Mike Guidry  guidry@utk.edu
//  December 9, 2001
// --------------------------------------------------------------------------------------------------------------------------


import java.awt.*;
import java.net.*;
import java.lang.*;

public class GraphicsGoodies2 extends Frame {

// Conversion factor from ln to log_10
public static final double log10 = 0.434294482;  // Base-10 log of e


    // ---------------------------------------------------------------------------------------------------------------
    //  Method rightSuperScript to position precisely a right superscript
    //  on a string in graphics mode.  ARGUMENTS:
    //     String s - The string to which the right superscript will be added
    //     String ss - The superscript to be added
    //     int x - The x coordinate in pixels for the main string
    //     int y - The y coordinate in pixels for the main string
    //     Font f - The font currently in use.  See most recent setFont(),
    //              or use getFont() method of Font object
    //     String relscale - Takes values "small", "medium", and "large"
    //                       and sets relative size of superscript relative
    //                       to main string.  These make the size of the
    //                       superscript 5,4, and 3 points smaller than
    //                       main string.  Default is "medium" (4 points smaller)
    //     Graphics g - The graphics object from which this method is being
    //                  called.  Typically set in something like the argument
    //                  of a paint method:
    //                       public void paint(Graphics g){
    //                          statements of method paint
    //                       }
    //                  from which this method is being called.
    //
    //     USAGE:
    //         GraphicsGoodies2 gg=new GraphicsGoodies2();  //Instantiate this class
    //         int leng=gg.rightSuperScript(s,ss,x,y,f,relscale,g);
    //
    //     The value leng returned is the length in pixels of the string with
    //     superscript appended.  This is useful for concatenation of further
    //     strings on the original string plus subscript.  Here is an example
    //     of typesetting a string with multiple superscripts that uses the
    //     returned string length to position subsequent portions of the string:
    //
    //  GraphicsGoodies2 gg=new GraphicsGoodies2();
    //  int len=0;
    //  len+=gg.rightSuperScript("M","1",xline,yline,font18,"medium",g);
    //  len+=gg.rightSuperScript("d","1",xline+len,yline,font18,"medium",g);
    //  len+=gg.rightSuperScript(" = M","2",xline+len,yline,font18,"medium",g);
    //  len+=gg.rightSuperScript("d","2",xline+len,yline,font18,"medium",g);
    // -------------------------------------------------------------------------------------------------------------------

    public int rightSuperScript(String s,String ss,int x,int y,Font f,String relscale,Graphics g){

        FontMetrics stringFontMetrics=getFontMetrics(f);
        g.setFont(f);                // Get the info on the main string font
        int fsize=f.getSize();
        String fname=f.getName();
        int style=f.getStyle();
        if (style==2){style=0;}  // Don't allow a superscript to be italic
        if (style==3){style=1;}  // or bold italic (not good style).
        int sizedecrease=4;      // Set size offset of superscript
        if (relscale=="small"){sizedecrease=5;}
        if (relscale=="large"){sizedecrease=3;}
        Font superf=new Font(fname, style, fsize-sizedecrease); // ss font
        FontMetrics superfFontMetrics=getFontMetrics(superf);
        int rightshift=stringFontMetrics.stringWidth(s);   // ss positioning
        int upshift=stringFontMetrics.getAscent();
        g.drawString(s,x,y);   //  Draw main string
        g.setFont(superf);
        int superwidth=superfFontMetrics.stringWidth(ss);
        g.drawString(ss,x+rightshift+1,y-upshift/3);  //  Draw superscript
        g.setFont(f);    // Reset font to original
        return rightshift+superwidth+1;   // Return full width of the
                                                        // string + superscript, to
                                                        // help in positioning subsequent
                                                        // strings.
     }



    // ------------------------------------------------------------------------------------------------------------------
    //  Method rightSubScript to position precisely a right subscript
    //  on a string in graphics mode.  ARGUMENTS:
    //     String s - The string to which the right subscript will be added
    //     String ss - The superscript to be added
    //     int x - The x coordinate in pixels for the main string
    //     int y - The y coordinate in pixels for the main string
    //     Font f - The font currently in use.  See most recent setFont(),
    //              or use getFont() method of Font object
    //     String relscale - Takes values "small", "medium", and "large"
    //                       and sets relative size of superscript relative
    //                       to main string.  These make the size of the
    //                       superscript 5,4, and 3 points smaller than
    //                       main string.  Default is "medium" (4 points smaller)
    //     Graphics g - The graphics object from which this method is being
    //                  called.  Typically set in something like the argument
    //                  of a paint method:
    //                       public void paint(Graphics g){
    //                          statements of method paint
    //                       }
    //                  from which this method is being called.
    //
    //     USAGE:
    //         GraphicsGoodies2 gg=new GraphicsGoodies2();  //Instantiate this class
    //         int leng=gg.rightSubScript(s,ss,x,y,f,relscale,g);
    //
    //     The value leng returned is the length in pixels of the string with
    //     superscript appended.  This is useful for concatenation of further
    //     strings on the original string plus subscript. Here is an example
    //     of typesetting a string with multiple subscripts that uses the
    //     returned string length to position subsequent portions of the string:
    //
    //  GraphicsGoodies2 gg=new GraphicsGoodies2();
    //  int len=0;
    //  len+=gg.rightSubScript("M","1",xline,yline,font18,"medium",g);
    //  len+=gg.rightSubScript("d","1",xline+len,yline,font18,"medium",g);
    //  len+=gg.rightSubScript(" = M","2",xline+len,yline,font18,"medium",g);
    //  len+=gg.rightSubScript("d","2",xline+len,yline,font18,"medium",g);
    //
    // ----------------------------------------------------------------------------------------------------------------------

    public int rightSubScript(String s,String ss,int x,int y,Font f,String relscale,Graphics g){
        FontMetrics stringFontMetrics=getFontMetrics(f);
        g.setFont(f);  // Get the info on the main string font
        int fsize=f.getSize();
        String fname=f.getName();
        int style=f.getStyle();
        if (style==2){style=0;}    // Don't allow a subscript to be italic
        if (style==3){style=1;}   // or bold italic (not good style).
        int sizedecrease=4;      // Set size offset of superscript
        if (relscale=="small"){sizedecrease=5;}
        if (relscale=="large"){sizedecrease=3;}
        Font superf=new Font(fname, style, fsize-sizedecrease); // ss font
        FontMetrics superfFontMetrics=getFontMetrics(superf);
        int rightshift=stringFontMetrics.stringWidth(s);   // ss positioning
        int upshift=stringFontMetrics.getAscent();
        g.drawString(s,x,y);   //  Draw main string
        g.setFont(superf);
        int superwidth=superfFontMetrics.stringWidth(ss);
        g.drawString(ss,x+rightshift+1,y+upshift/3);  //  Draw superscript
        g.setFont(f);    // Reset font to original
        return rightshift+superwidth+1;   // Return full width of the
                                                        // string + superscript, to
                                                        // help in positioning subsequent
                                                        // strings.
     }



    // -------------------------------------------------------------------------------------------------------------------
    //  Method decimalPlace returns string  representation of double
    //  with a fixed number of places after the decimal.  The number of places
    //  after the decimal is given by integer "nright" (>=0) and the double is
    //  passed as the variable "number".  Routine handles both decimal and
    //  scientific (E) notation.  Rounds floating point style (e.g., 5.676
    //  truncated to 2 decimal places returns 5.68, not 5.67).  Pads the right
    //  with zeros if insufficient digits after the decimal (e.g., a request
    //  to truncate 5.67 to 4 decimal places returns 5.6700).  If nright=0, no
    //  decimal is shown (e.g., 3 instead of 3.).
    //
    //  EXAMPLE OF USING FROM ANOTHER CLASS:
    //       GraphicsGoodies2 gg=new GraphicsGoodies2();  // Instantiate this class
    //       String nstring = gg.decimalPlace(nright,number);
    //       gg.drawString("Variable=" + decimalPlace(3,variable),100,200);
    //
    // -------------------------------------------------------------------------------------------------------------------


    public String decimalPlace(int nright, double number) {
        double n=number;
        String tleft;             // Mantissa left of .
        String tright;           // Original mantissa right of .
        String tright2="";    // Final mantissa right of .
        String eleft="";
        String eright="";

        String total;
        total=String.valueOf(n);
        int temp1=0;
        int temp2=0;
        int i=1;
        int dotil=0;
        int nperiod=0;

        //  Check for scientific notation
        int ne=total.indexOf("E");
        if(ne > -1){
            eleft=total.substring(0,ne);
            eright=total.substring(ne);   //  string containing exponent
            total=eleft;                        // string containing mantissa
        }

        //  Roundoff to proper number of places.  Last digit retained
        //  bumped up by one if the first one cut off is 5 or greater.
        Double mydouble=Double.valueOf(total);  // 2 steps to convert string to double
        double nn=mydouble.doubleValue(); 
        double nnn=Math.round(nn*Math.pow(10,nright));
        total=String.valueOf(nnn/Math.pow(10,nright));

        //  Split mantissa left of the decimal place;
        //  return if no decimal or no places to right of decimal
        nperiod=total.indexOf(".");
        if(nperiod == 0 || nperiod == -1){return total+eright;}
        tleft=total.substring(0,nperiod);    // mantissa left of decimal
        tright=total.substring(nperiod);     // original mantissa right of . (including decimal)

        //  Pad tright with zeros if necessary to bring up
        //  to the desired number of places to right of decimal
        if(tright.length()-1 <= nright){
            dotil=nright-tright.length();
            for (i=0; i<= dotil+1; i++){
                tright=tright+"0";
            }
        }

        //  truncate the mantissa to right of decimal to nright places
        temp1=0;
        temp2=nright+1;
        if(tright.length() > nright) {
            try{tright2=tright.substring(temp1,temp2);}
                catch (StringIndexOutOfBoundsException e)
            { ; }
        }
        else {
            tright2=tright;
        }
        // If number of decimal places is zero, strip off any decimal
        // (e.g., return 3 instead of 3.)
        if (nright == 0){
            tright2 = tright2.substring(1,tright2.length());
        }
        //  Return the truncated string
        return tleft+tright2+eright;
    }


    // -----------------------------------------------------------------------------------------------------------------------
    //  Method drawVector to draw a vector in graphics mode.  Method is overloaded
    //  (two versions of it).  This is the cartesian argument method.  The next
    //  method below defines the polar coordinate method.  Java decides which to
    //  use by examining the data types of the call arguments.
    //  ARGUMENTS FOR THIS (CARTESIAN) METHOD:
    //     int x1 - The x coordinate in pixels for the beginning of vector
    //     int y1 - The y coordinate in pixels for the beginning of vector
    //     int x2 - The x coordinate in pixels for the arrow tip of vector
    //     int y2 - The y coordinate in pixels for the arrow tip of vector
    //     Graphics g - The graphics object from which this method is being
    //                  called.  Typically set in something like the argument
    //                  of a paint method:
    //                       public void paint(Graphics g){
    //                          statements of method paint
    //                       }
    //                  from which this method is being called.
    //
    //   USAGE:
    //         GraphicsGoodies2 gg=new GraphicsGoodies2(); // Instantiate this class
    //         gg.drawVector(x1,y1,x2,y2,g);               // Access the method
    //
    // ------------------------------------------------------------------------------------------------------------------------

    // cartesian version of the method

    public void drawVector(int x1,int y1,int x2,int y2,Graphics g) {
        int fac=1;
        double theta=0.38;  // angle for shape of arrowhead
        double sintheta=Math.sin(theta);
        double dely=(double)(y2-y1);
        double delx=(double)(x2-x1);
        double phi=0.0;

        if(Math.abs(delx) <= 1.0e-10){
            phi=1.5708;         // handle exactly vertical line
        }
        else{
            phi=Math.atan(dely/delx);
        }
        double a=4;  // Sets arrowhead halfwidth in pixels
        if(x2 < x1){fac=-1;}  // For upper left and lower left quadrants
        int delxR=fac*(int)(a*Math.cos(phi-theta)/sintheta);
        int delyR=fac*(int)(a*Math.sin(phi-theta)/sintheta);
        int delxL=fac*(int)(a*Math.cos(phi+theta)/sintheta);
        int delyL=fac*(int)(a*Math.sin(phi+theta)/sintheta);
        int[] xharrow={x2,x2-delxL,x2-delxR};
        int[] yharrow={y2,y2-delyL,y2-delyR};
        g.drawLine(x1,y1,x2,y2);
        g.fillPolygon(xharrow,yharrow,3);
    }


    // ---------------------------------------------------------------------------------------------------------------------
    //  Method drawVector to draw a vector in graphics mode.  Method is overloaded
    //  (two versions of it).  This is the polar coordinate method.  The preceding
    //  method above defines the cartesian method.  Java decides which to
    //  use by examining the data types of the call arguments.
    //  ARGUMENTS FOR THIS (POLAR) METHOD:
    //     int x1 - The x coordinate in pixels for the beginning of vector
    //     int y1 - The y coordinate in pixels for the beginning of vector
    //     int r - The length in pixels for the vector
    //     double phi - The angle in degrees for the orientation of the vector.
    //                  Uses the Java angle convention:  Zero degrees at
    //                  the 3 o'clock position and angles measured counterclockwise.
    //                  Thus, for a vector straight up phi=90 degrees and for
    //                  one straight down phi= -90 degrees.
    //     Graphics g - The graphics object from which this method is being
    //                  called.  Typically set in something like the argument
    //                  of a paint method:
    //                       public void paint(Graphics g){
    //                          statements of method paint
    //                       }
    //                  from which this method is being called.
    //
    //   USAGE:
    //         GraphicsGoodies2 gg=new GraphicsGoodies2(); // Instantiate this class
    //         gg.drawVector(x1,y1,r,phi,g);               // Access the method
    //
    // --------------------------------------------------------------------------------------------------------------------

    // the polar coordinate method

    public void drawVector(int x1,int y1,int r,double phi,Graphics g) {
        phi=-phi/57.2;     // Convert to radians & insert minus sign to account
                                // for java phase convention:  angles measured
                                // counterclockwise from the 3 o'clock position.
        double delx=Math.cos(phi)*(double)r;
        double dely=Math.sin(phi)*(double)r;
        int x2=x1+(int)delx;
        int y2=y1+(int)dely;
        this.drawVector(x1,y1,x2,y2,g);  // Invoke the other (cartesian) method

    }

    // ------------------------------------------------------------------------------------------------------------------------------
    //  Method drawDashedLine draws a dashed line in a graphics environment.  Method is
    //  overloaded (two versions of it).  This one allows the length of the dash and the
    //  blank to be specified through deldash and delblank.  The other does not take these
    //  arguments and uses defaults for them.  Arguments have similar explanation as
    //  for drawVector above.
    // ------------------------------------------------------------------------------------------------------------------------------

    public void drawDashedLine(int x1,int y1,int x2,int y2,int deldash,int delblank,Graphics g) {
        double delx=(double)(x2-x1);
        double dely=(double)(y2-y1);
        double phi;
        if(Math.abs(delx) <= 1.0e-10){    //  Handle the vertical line case
            phi=1.5708;
        } else {
            phi=Math.atan(dely/delx);
        }
        double cosphi=Math.cos(phi);
        double sinphi=Math.sin(phi);
        double h;
        if(Math.abs(sinphi) <= 1.0e-10){  //  Handle the horizontal line case
            h=delx;
        } else {
            h=dely/sinphi;
        }
        int dashnumber=(int)(h/(double)(deldash+delblank)+0.5);
        double delxdash=(double)deldash*cosphi;
        double delydash=(double)deldash*sinphi;
        double delxblank=(double)delblank*cosphi;
        double delyblank=(double)delblank*sinphi;
        double oldx=(double)x1;
        double oldy=(double)y1;
        double newx=(double)x1+delxdash;
        double newy=(double)y1+delydash;
        for (int i = 1; i <= dashnumber; i++){
            g.drawLine((int)oldx,(int)oldy,(int)newx,(int)newy);
            oldx+=delxdash+delxblank;
            oldy+=delydash+delyblank;
            newx=oldx+delxdash;
            newy=oldy+delydash;
        }
    }

    //  Overloaded version that does not expect the length of the dash and blanks
    //  to be specified by user.  See further explanation above.

    public void drawDashedLine(int x1,int y1,int x2,int y2,Graphics g) {
        int deldash=4;
        int delblank=4;
        this.drawDashedLine(x1,y1,x2,y2,deldash,delblank,g);
    }

    // --------------------------------------------------------------------------------------------------------------------------
    //  Method drawDottedLine draws a dotted line between the specified endpoints.  See
    //  the discussion of the method drawVector above for the arguments.
    // --------------------------------------------------------------------------------------------------------------------------


    public void drawDottedLine(int x1,int y1,int x2,int y2,Graphics g) {
    int deldash=1;
    int delblank=3;
    this.drawDashedLine(x1,y1,x2,y2,deldash,delblank,g);
    }



    // ---------------------------------------- METHOD plotIt ---------------------------------------------

    // Method plotIt to draw x-y plot for arbitrary number of curves in
    // linear-linear, log-linear, or log-log form.  Set up to
    // to be called from another graphics object, permitting a plot to be
    // inserted on an arbitrary part of the canvas.  (Thus the user
    // is responsible to be certain that the calling graphics object does
    // not erase the plot generated by this method with its own repaints.) Data
    // are passed to the method as arrays, as are many parameters that give
    // detailed control over the look of the resulting plot.  The user must
    // set up the parameters and the data arrays to be passed from the program
    // invoking the method; an example is given below.  PlotIt assumes data
    // equally spaced in the x coordinate. PlotIt will work with either
    // standalone applications or applets.
    //
    // (See the programs plotItTest.java and ggtest.java in this directory
    // for examples of using plotIt in an application and an applet respectively.)
    //
    //     USAGE:
    //
    //     The invocation of the GraphicsGoodies2.plotIt(...) method
    //     will generally be in a paint method of say a Panel or
    //     Applet object, as illustrated below.  First instantiate
    //     the GraphicsGoodies2 class and then use the instance to
    //     invoke the plotIt method from within the paint method:
    //
    //    GraphicsGoodies2 gg = new GraphicsGoodies2();  // Instantiate
    //
    //    public void paint(Graphics g){
    //
    //        // invoke the GraphicsGoodies2.plotIt method
    //
    //        gg.plotIt(plotmode,x1,y1,x2,y2,kmax,imax,mode,
    //                  xlegoff,ylegoff,xdplace,ydplace,
    //                  npoints,doscalex,doscaley,doplot,xmin,xmax,ymin,ymax,
    //                  delxmin,delxmax,delymin,delymax,
    //                  lcolor,x,y,bgcolor,axiscolor,legendfg,framefg,
    //                  dropShadow,legendbg,labelcolor,ticLabelColor,
    //                  xtitle,ytitle,curvetitle,logStyle, ytickIntervals,
    //                  xtickIntervals, showLegend, g);
    //     }
    //
    //     Where: Graphics g - The graphics object from which this method is being
    //            called.  Other parameters are described below.
    //
    //
    //  LIST OF PARAMETERS WITH EXAMPLE VALUES AND DESCRIPTION:
    //
    //    int plotmode=1;       // Type of plot: 0=linear, 1=log-lin, 2=log-log
    //
    //    // Set the coordinates (relative to the graphics container
    //    // from which the method is being invoked) for the plot
    //    // window. Coordinates in pixels measured from the upper
    //    // left corner of the container (e.g., a frame or an applet window).
    //
    //    int x1=5;             // x coordinate of upper left corner of plot
    //    int y1=5;             // y coordinate of upper left corner of plot
    //    int x2=215;           // x coordinate of lower right corner of plot
    //    int y2=265;           // y coordinate of lower right corner of plot
    //
    //    int kmax=200;         // Max number of points for each curve. Must
    //                          // be as large as largest entry in npoints[].
    //    int imax=2;           // Max number of separate curves
    //
    //    int mode[]={6,2};     // array giving display mode for curve
    //                          // with 1=solid line, 2=lines+dots, 3=dashed line,
    //                          // 4=open circles, 5=filled circles, 6=filled squares,
    //                          // 7=open squares, 8=x, 9=x+square, 10=+, 11=+ and square,
    //                          // 12=open diamonds, 13=open down triangle,
    //                          // 14=open up triangle, 15=horizontal open oval,
    //                          // 16=horizontal filled oval, 17=vertical open oval,
    //                          // 18=vertical filled oval.
    //                          // If mode other than line mode, dotSize below sets
    //                          // the size of the plotting symbol.
    //
    //    int dotSize = 3;      // If plotting dots or symbols, width of dot
    //                          // in pixels.  Minimum size is 3 pixels.
    //                          // Smaller sizes will be reset to 3 pixels.
    //
    //    // Offset of legend box from upper left corner of plot:
    //
    //    int xlegoff=100;      // x offset in pixels from upper left of frame
    //    int ylegoff=10;       // y offset in pixels from upper left of frame
    //
    //    int xdplace=0;        // Number decimal places for numbers on x axis
    //    int ydplace=0;        // Number decimal places for numbers on y axis
    //
    //    int npoints[]={20,100};  // vector giving number of data points for curve
    //                             // i with i=0,1, ... imax.  Entries less than
    //                             // or equal to kmax.  Rule of thumb: if display
    //                             // mode is lines (mode[i]=2), npoints[i] should be
    //                             // about 1/2 the graph width in pixels.  If mode
    //                             // is dots (e.g., mode[i]=1), npoints[i] should be
    //                             // about 10-15% of graph width in pixels.  If
    //                             // display mode is dashed line (mode[i]=7),
    //                             // npoints[i] should be about 1/2-3/4 graph width.
    //
    //    int doscalex=0;   // 1 -> scale min & max of x by data; 0 -> no autoscale
    //    int doscaley=0;   // 1 -> scale min & max of y by data; 0 -> no autoscale
    //
    //    int doplot[] = {1,1};    // Vector controlling whether curve
    //                             // i=0,1,2,...imax is plotted: plot=1; noplot=0
    //                             // (Reset to toggle curve visibility)
    //
    //    // Following overridden by autoscaling if doscalex=1 or doscaley=1
    //
    //    double xmin=0;           // Min plot x if doscalex=0
    //    double xmax=10.0;        // Max plot x if doscalex=0
    //    double ymin=1.0;         // Min plot y if doscaley=0
    //    double ymax=10000.0;     // Max plot y if doscaley=0
    //
    //    // Set the amount of empty space above, below, right, left
    //    // of the plotted data as a fraction of the total width of
    //    // the plot area.  For example, delymax=0.10 leaves 10%
    //    // empty space above the highest data point
    //
    //    double delxmin=0.0;      // fraction space left on left
    //    double delxmax=0.0;      // fraction space left on right
    //    double delymin=0.0;      // fraction space left below
    //    double delymax=0.0;      // fraction space left above
    //
    //    // Set the colors for the lines and symbols for the curves
    //
    //    Color lcolor[]={AIorange,AIpurple};       // imax entries
    //
    //    // Set the colors for the plot background, labels, & axes,
    //    // and legend box
    //
    //    Color bgcolor=gray250;            // plot background color
    //    Color axiscolor=gray51;           // axis color
    //    Color legendfg=Color.white;       // legend box color
    //    Color framefg=gray204;            // frame color
    //    Color dropShadow = gray153;       // legend box dropshadow color
    //    Color legendbg=gray204;           // legend box frame color
    //    Color labelcolor = gray51;        // axis label color
    //    Color ticLabelColor = gray51;     // axis tic label color
    //
    //  // Set the strings for labeling the axes and for
    //  // each of the curves in the plot legend
    //
    //    String xtitle="Time (weeks)";     // String for x-axis label
    //    String ytitle="$";                // String for y-axis label
    //    String curvetitle[]={"Salary", "Expenses"};        // imax entries
    //
    //    int logStyle = 0;     // 0 to show number, 1 to show log of number
    //                          // on axis when plot is logarithmic
    //
    //    int ytickIntervals = 4;     // Number of intervals between y ticks
    //    int xtickIntervals = 5;     // Number of intervals between x ticks
    //    boolean showLegend = true;  // Show legend box (true or false)
    //
    //
    //    SUPPLYING DATA TO THE PLOTTING METHOD:
    //
    //    You must supply data to the plotIt method, first by defining
    //    2-dimensional arrays to hold x and y values, and then filling
    //    those arrays with your data.  The following examples illustrate
    //    setting up the arrays and filling them.
    //
    //    // Create the data arrays.  They will be filled in the putData()
    //    // method defined below
    //
    //    double x[][]=new double[imax][kmax];  // data array for x coordinates
    //    double y[][]=new double[imax][kmax];  // data array for y coordinates
    //                                          // Form: x[i][k] and y[i][k], with
    //                                // i=0,1, ... imax = index of separate curves;
    //                                // k=0,1, ... kmax = index of data points
    //
    //
    //    // Sample routine to fill data arrays.  Replace this with a method
    //    // to fill the data arrays with the data you desire to plot.  Be
    //    // certain that the number of points generated is consistent with
    //    // npoints[] defined above.  Also note that PlotIt() expects
    //    // x coordinates of the data set to be equally spaced for best results
    //    // if plotting in the line mode (mode[i]=1).
    //
    //    void putData () {
    //	    // Fill the data array for plot number 1
    //	    for (int k=0; k<npoints[0]; k++){
    //            // First curve
    //	        x[0][k]=((xmax-xmin)/(double)npoints[0])*(double)k;
    //	        y[0][k]=350-2*x[0][k]*x[0][k];
    //            }
    //            // Second curve
    //	    for (int k=0; k<npoints[1]; k++){
    //	        x[1][k]=((xmax-xmin)/(double)npoints[1])*(double)k;
    //	        y[1][k]= 50 + 4.9*x[1][k]*x[1][k];
    //	    }
    //    }
    //
    //   See the programs plotItTest.java and ggtest.java in this directory
    //   for examples of using plotIt in an application and an applet respectively.
    // -------------------------------------------------------------------------------------------------------------------

    public void plotIt(int plotmode,int x1,int y1,int x2,int y2,int kmax,
        int imax,int mode[],int dotSize,int xlegoff,
        int ylegoff,int xdplace,int ydplace,
        int npoints[],int doscalex,int doscaley,int doplot[],
        double xxmin,double xxmax,double yymin,double yymax,
        double delxmin,double delxmax,double delymin,double delymax,
        Color lcolor[],Color bgcolor,
        Color axiscolor,Color legendfg,Color framefg,
        Color dropShadow,Color legendbg,Color labelcolor,
        Color ticLabelColor,String xtitle,String ytitle,
        String curvetitle[],int logStyle, int ytickIntervals,
        int xtickIntervals,boolean showLegend,
        double xx[][],double yy[][],Graphics g){

        double xmin, xmax, ymin, ymax;
        double x[][],y[][];

        int maxp;
        int maxhp;

        // Create memory for data arrays.  First argument is
        // the plot number.  Second argument is the data point
        // number for that plot.

        x=new double[imax][kmax];
        y=new double[imax][kmax];

        // Define the fonts to be used and their fontmetrics

        Font titleFont = new java.awt.Font("Arial", Font.PLAIN, 10);
        FontMetrics titleFontMetrics = getFontMetrics(titleFont);
        Font titleItalicFont = new java.awt.Font("Arial", Font.ITALIC, 10);
        FontMetrics titleItalicFontMetrics = getFontMetrics(titleFont);
        Font smallFont = new java.awt.Font("Arial", Font.PLAIN, 9);
        FontMetrics smallFontMetrics = getFontMetrics(smallFont);   
        Font smallsmallFont = new java.awt.Font("Arial", Font.PLAIN, 8);
        FontMetrics smallsmallFontMetrics = getFontMetrics(smallsmallFont);
        
        // reset values of arrays if log or log-log flags set

        int i,k;
        for (i=0; i<imax; i++){
            for(k=0; k<npoints[i]; k++){

		// Fix so if x < 0 or y <=0 it doesn't stop log plot
		if(xx[i][k]<=0.0){
			System.out.println("At x=" +decimalPlace(4, xx[i][k]) 
				+ " y["+i+"]="+ yy[i][k]+", Set x=1e-25 to avoid log plot crash ");
			xx[i][k] = 1e-25;
		}
		if(yy[i][k]<=0.0){
			System.out.println("At x=" +decimalPlace(4, xx[i][k]) 
				+ " y["+i+"]="+ yy[i][k]+", Set y=1e-25 to avoid log plot crash ");
			yy[i][k] = 1e-25;
		}

                switch(plotmode){
                    case 0:                 // linear plot
                    x[i][k]=xx[i][k];
                    y[i][k]=yy[i][k];

                    break;

                    case 1:                 // log-lin plot
                    x[i][k]=xx[i][k];

                    // first check whether possible to take Log of y
                    // and bail out if there are negative or zero numbers

                    if(yy[i][k]<=0.0){
                        g.setFont(titleFont);
                        g.setColor(axiscolor);
                        g.drawString("Data for y zero or negative.",x1+50,y1+20);
                        g.drawString("Can't make Log plot.",x1+50,y1+35);
                        g.drawRect(x1,y1,x2-x1,y2-y1);
                        System.out.println
                          ("y-data zero or negative. No can do Log plot: "
                              +"x=" +xx[i][k] + " y="+ yy[i][k]);
                        // getAppletContext().showStatus  // Valid only for applet
                        //  ("Data Neg. or Zero; can't plot Log");
                        return;
                    }

                    y[i][k]=log10*Math.log(yy[i][k]);
                    break;

                    case 2:                 // log-log plot

                    // first check whether possible to take Log of x & y
                    // and bail out if there are negative or zero numbers

                    if(yy[i][k]<=0.0 || xx[i][k]<= 0.0){
                        g.setFont(titleFont);
                        g.setColor(axiscolor);
                        g.drawString("Data values zero or negative.",x1+50,y1+20);
                        g.drawString("Can't make Log-Log plot.",x1+50,y1+35);
                        g.drawRect(x1,y1,x2-x1,y2-y1);
                        System.out.println
                          ("Data zero or negative. No can do Log-Log plot: "
                              +"x=" +xx[i][k] + " y="+ yy[i][k]);
                        // getAppletContext().showStatus  // Valid for applet only
                        //   ("Data Neg. or Zero; can't plot Log-Log");
                        return;
                    }
                    x[i][k]=log10*Math.log(xx[i][k]);
                    y[i][k]=log10*Math.log(yy[i][k]);
                    break;
                }

             }
        }

        // Reset min and max values to their base-10 logs if log modes
        if(plotmode == 1 || plotmode == 2) {
            yymin = log10*Math.log(yymin);
            yymax = log10*Math.log(yymax);
        }
        if(plotmode == 2) {
            xxmin = log10*Math.log(xxmin);
            xxmax = log10*Math.log(xxmax);
        }

        // find mimima and maxima in data set
        if(doscalex==0){         // set default min & max values for x
            xmin=xxmin;
            xmax=xxmax;
        }
        else{
            xmin=x[0][0];
            xmax=x[0][0];
            for (i=0; i<imax; i++){
                for(k=0; k<npoints[i]; k++){
                    if(x[i][k] < xmin){xmin=x[i][k];}
                    if(x[i][k] > xmax){xmax=x[i][k];}
                }
            }
        }

        if(doscaley==0){           // set default min & max values for y
            ymin=yymin;
            ymax=yymax;
        }
        else{
            ymin=y[0][0];
            ymax=y[0][0];
            for (i=0; i<imax; i++){
                for(k=0; k<npoints[i]; k++){
                    if(y[i][k] < ymin){ymin=y[i][k];}
                    if(y[i][k] > ymax){ymax=y[i][k];}
                }
            }
        }

        // Set the amount of empty space above, below, right, left
        // of the plotted data

        ymin=ymin-delymin*Math.abs(ymax-ymin);
        ymax=ymax+delymax*Math.abs(ymax-ymin);
        xmin=xmin-delxmin*Math.abs(xmax-xmin);
        xmax=xmax+delxmax*Math.abs(xmax-xmin);

        // Set some sizes
        int psize=Math.max(3,dotSize);    // point symbol size;
                                                         // must be integer >= 3
        int hpsize=psize/2;
        int wid=x2-x1;          // width of plot in pixels
        int hite=y2-y1;         // height of plot in pixels
        int xoffset=65;         // pixels to left of y axis
        int yoffset=40;         // pixels below x axis
        int topmarg=15;        // pixels above graph
        int rightmarg=25;      // pixels to right of graph

        double xscale,yscale;

        // draw plot rectangle
        g.setColor(bgcolor);
        g.fillRect(x1,y1,wid,hite);
        g.setColor(framefg);
        g.drawRect(x1,y1,wid,hite);
        g.setColor(axiscolor);

        // draw plot axes
        g.drawLine(x1+xoffset,y2-yoffset,x1+xoffset,y1+topmarg);
        g.drawLine(x1+xoffset,y2-yoffset,x1+wid-rightmarg,y2-yoffset);

        // set scaling factors for x,y -> pixels
        xscale=1.0*(wid-xoffset-rightmarg)/(Math.abs(xmax-xmin));
        yscale=1.0*(hite-yoffset-topmarg)/(ymax-ymin);

      //  Tic marks for x axis
      double ticspace=(xscale*Math.abs(xmax-xmin)/(double) xtickIntervals);
      double tmark;
      int fshift, vshift;
      g.setFont(smallFont);

          for (k=0; k<=xtickIntervals; k++){
            // Tic marks
            g.setColor(axiscolor);
            g.drawLine(x1+xoffset+(int)(k*ticspace), y2-yoffset,
                x1+xoffset+(int)(k*ticspace), y2-yoffset-5);
            // Labels for ticmarks
            g.setColor(ticLabelColor);
            tmark=(double)(k*ticspace)/xscale+xmin;
            fshift=smallFontMetrics.stringWidth (this.decimalPlace(xdplace,tmark))/2;

            if(plotmode<2)     //  If not log-log
              {
                 g.drawString(this.decimalPlace(xdplace,tmark),
                    x1+xoffset+(int)(k*ticspace)-fshift, y2-yoffset+15);
               }
            else                      //  For log-log case
              {
                 // Decide whether to display number or log
                 if (logStyle == 0 && plotmode == 2) {
                     tmark = Math.pow(10,tmark);
                 }

                // Following quick fix causes zero rather than something like
                // 9e-16 to be printed at the y-origin

                if(Math.abs(tmark) < 1e-10) tmark=0;

                 fshift=smallFontMetrics.stringWidth(this.decimalPlace(xdplace,tmark))/2;
                  g.drawString(this.decimalPlace(xdplace,tmark),
                    x1+xoffset+(int)(k*ticspace) - fshift, y2-yoffset+15);
               }
          }

        //  Tic marks for y axis
        ticspace=(yscale*Math.abs(ymax-ymin)/(double)ytickIntervals);
        for (k=0; k<=ytickIntervals; k++){
            // Tic marks
            g.setColor(axiscolor);
            //if(k != ytickIntervals){
                g.drawLine(x1+xoffset, y2-yoffset-(int)(k*ticspace),
                        x1+xoffset+5, y2-yoffset-(int)(k*ticspace));
            // }
            // Labels for tic marks
            g.setColor(ticLabelColor);
            tmark=(double)(k*ticspace)/yscale;
            vshift=smallFont.getSize()/2;
            double yvalue=ymin + (double)k*Math.abs(ymax-ymin)/(double)ytickIntervals;
            // If log plot, decide whether to display number or log
            if (logStyle == 0 && plotmode >= 1) {
                yvalue = Math.pow(10,yvalue);
            }

            // Following quick fix causes zero rather than something like
            // 9e-16 to be printed at the y-origin

            if(Math.abs(yvalue) < 1e-10) yvalue=0;

            fshift=smallFontMetrics.stringWidth(this.decimalPlace(ydplace,yvalue));

            g.drawString(this.decimalPlace(ydplace,yvalue),
            x1+xoffset-fshift-5,
            y2-yoffset-(int)(k*ticspace)+vshift);
        }

        // Date, comments, subversion (SVN) revision numbers, and parameters

        g.setFont(smallFont);
        int tdsy = y2+22;
        if(!StochasticElements.longFormat && !StochasticElements.amPlottingRates) tdsy -= 12;
        int tdsx = 40;
        int pspacer = smallFont.getSize()+3;
		
		String tsteps = "|Int steps="+(StochasticElements.totalTimeSteps 
				- StochasticElements.totalTimeStepsZero);
		String plsteps = "|Plot steps="+StochasticElements.nintervals;
		String vnumstuff = "";
		if(StochasticElements.showSVNversion) vnumstuff = 
			StochasticElements.replaceWhiteSpace(StochasticElements.vSVN,"");
		vnumstuff = vnumstuff + plsteps + tsteps + StochasticElements.os5;

        g.drawString(StochasticElements.os1,x1+tdsx,tdsy);
        g.drawString(StochasticElements.os2,x1+tdsx,tdsy+pspacer);	
        g.drawString(StochasticElements.os3,x1+tdsx,tdsy+2*pspacer);
        g.drawString(vnumstuff,x1+tdsx,tdsy+3*pspacer);
        //if(StochasticElements.showSVNversion){
        //    g.drawString(StochasticElements.vSVN+plsteps+tsteps+" "
        //    		+StochasticElements.os5,x1+tdsx,tdsy+3*pspacer);
        //}

        //  Label for x-axis
        g.setFont(titleFont);
        g.setColor(labelcolor);
        if(plotmode==2 && logStyle == 1){xtitle= "Log "+xtitle;}
        fshift=titleFontMetrics.stringWidth(xtitle)/2;
        g.drawString(xtitle, x1+xoffset/2+wid/2-rightmarg/2-fshift, y2-8);
        
        //  Label for y axis
        String ss1 = ytitle.substring(0,1);
        String ss2 = ytitle.substring(1);
        if(plotmode>0 && logStyle == 1){ytitle="Log ";}
        vshift=titleFont.getSize()/2;
        g.drawString(ytitle, x1+xoffset+12, y1+topmarg+vshift-7);
        g.setFont(titleItalicFont);
        g.drawString(ss1,x1+xoffset+12+titleItalicFontMetrics.stringWidth("Log "), y1+topmarg+vshift-7);
        g.setFont(titleFont);
        g.drawString(ss2 ,x1+xoffset+12+titleItalicFontMetrics.stringWidth("Log  "+ss1), y1+topmarg+vshift-7);
        
        // plot the graphs
        int xk, yk, xprev, yprev;
	   
        // To allow points on the boundary to be plotted, slightly increase
        // the bounding box size that will be used below to exclude points
        // outside the bounds of the plot.

        double xminex = xmin - 0.01*Math.abs(xmin-xmax);
        double xmaxex = xmax + 0.01*Math.abs(xmin-xmax);
        double yminex = ymin - 0.01*Math.abs(ymin-ymax);
        double ymaxex = ymax + 0.01*Math.abs(ymin-ymax);

        Color eplusColor = new Color(153,204,153);    // Color for pos energy
        Color eminusColor = new Color(255,153,0);     // Color for neg energy

        if(StochasticElements.plotEnergy){
                
            // Handle energy plot separately so that color can be switched between
            // positive and negative values for the plotted absolute value.
        
            for(k=0; k<npoints[imax-1]; k++){
                if(StochasticElements.plotdE){                        // If dE
                    if(StochasticElements.deNow[k] >= 0){
                            g.setColor(eplusColor);
                    } else {
                            g.setColor(eminusColor);
                    }
                } else {                                                        // If E
                    if(StochasticElements.eNow[k] >= 0){
                            g.setColor(eplusColor);
                    } else {
                            g.setColor(eminusColor);
                    }
                }
        
                // Don't plot if out of bounds
                if(x[imax-1][k] < xminex || x[imax-1][k] > xmaxex || y[imax-1][k] < yminex  
                    || y[imax-1][k] > ymaxex){continue;}
                xk=(int)((x[imax-1][k]-xmin)*xscale);
                yk=(int)((y[imax-1][k]-ymin)*yscale);
                // Use + symbol plot for E or dE
                g.drawLine(x1+xoffset+xk-hpsize,
                    y2-yk-yoffset,
                    x1+xoffset+xk-hpsize+psize,
                    y2-yk-yoffset);
                g.drawLine(x1+xoffset+xk,
                    y2-yk-hpsize-yoffset,
                    x1+xoffset+xk,
                    y2-yk-hpsize-yoffset+psize);
            }                          
        }
		   
        // Now plot the isotope curves.  Loop over curves:

        for(i=0; i<imax-1; i++){
            int kindex = 0;
            if(doplot[i]==1){
                g.setColor(lcolor[i]);

                switch(mode[i]) {

                    case 1:                     // If solid line plot

                        xk=(int)((x[i][0]-xmin)*xscale);
                        yk=(int)((y[i][0]-ymin)*yscale);
                        for(k=1; k<npoints[i]; k++) {
                            xprev=xk;
                            yprev=yk;
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}

                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            if( xprev < 0 || yprev < 0 ){
                                continue;   // Off plot
                            }

                            if(yk<0 || xk < 0){continue;}  // Off plot
                            g.drawLine(xprev+xoffset+x1,
                                        -yprev+y2-yoffset,
                                        xk+xoffset+x1,
                                        -yk+y2-yoffset);
                        }
                        break;

                    case 2:                  // If line + point

                        xk=(int)((x[i][0]-xmin)*xscale);
                        yk=(int)((y[i][0]-ymin)*yscale);
                        for(k=1; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xprev=xk;
                            yprev=yk;
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            if( xprev < xmin || yprev < ymin ){
                            continue;   // Off plot
                            }
                            g.drawLine(xprev+xoffset+x1,
                                        -yprev+y2-yoffset,
                                        xk+xoffset+x1,
                                        -yk+y2-yoffset);
                            g.fillOval(x1+xoffset+xk-hpsize,
                                        y2-yk-hpsize-yoffset,
                                        psize,psize);
                        }
                        break;

                    case 3:                  // If dashed line plot

                        xk=(int)((x[i][0]-xmin)*xscale);
                        yk=(int)((y[i][0]-ymin)*yscale);
                        for(k=1; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xprev=xk;
                            yprev=yk;
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            if( xprev < xmin || yprev < ymin ){
                            continue;   // Off plot
                            }
                            kindex ++;
                            if(kindex == 2){ // blank every other segment
                                kindex = 0;
                                g.setColor(bgcolor);
                            } else {
                                g.setColor(lcolor[i]);
                            }
                            g.drawLine(xprev+xoffset+x1,
                                        -yprev+y2-yoffset,
                                        xk+xoffset+x1,
                                        -yk+y2-yoffset);
                        }
                        break;

                    case 4:                 //  If open circle plot
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            g.drawOval(x1+xoffset+xk-hpsize,
                                        y2-yk-hpsize-yoffset,
                                        psize,psize);
                        }
                        break;

                    case 5:                 //  If solid circle plot
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            g.fillOval(x1+xoffset+xk-hpsize,
                                        y2-yk-hpsize-yoffset,
                                        psize,psize);
                        }
                        break;

                    case 6:                 // If filled square plot
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            g.fillRect(x1+xoffset+xk-hpsize,
                                        y2-yk-hpsize-yoffset,
                                        psize,psize);
                        }
                        break;

                    case 7:                 // If open square plot
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            g.drawRect(x1+xoffset+xk-hpsize,
                                        y2-yk-hpsize-yoffset,
                                        psize,psize);
                        }
                        break;

                    // Following is poor man's dashed line.  It
                    // leaves equal blank intervals in the x
                    // coordinate, not in the local length of the
                    // line segement.

                    case 8:                 // x-symbol plot
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            g.drawLine(x1+xoffset+xk-hpsize,
                                        y2-yk-hpsize-yoffset,
                                        x1+xoffset+xk-hpsize+psize,
                                        y2-yk-hpsize-yoffset+psize);
                            g.drawLine(x1+xoffset+xk-hpsize,
                                        y2-yk-hpsize-yoffset+psize,
                                        x1+xoffset+xk-hpsize+psize,
                                        y2-yk-hpsize-yoffset);
                        }
                        break;

                    case 9:                 // x-symbol plot + square
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            g.drawRect(x1+xoffset+xk-hpsize,
                                        y2-yk-hpsize-yoffset,
                                        psize,psize);
                            g.drawLine(x1+xoffset+xk-hpsize,
                                        y2-yk-hpsize-yoffset,
                                        x1+xoffset+xk-hpsize+psize,
                                        y2-yk-hpsize-yoffset+psize);
                            g.drawLine(x1+xoffset+xk-hpsize,
                                        y2-yk-hpsize-yoffset+psize,
                                        x1+xoffset+xk-hpsize+psize,
                                        y2-yk-hpsize-yoffset);
                        }
                        break;

                    case 10:                 // + symbol plot
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            g.drawLine(x1+xoffset+xk-hpsize,
                                        y2-yk-yoffset,
                                        x1+xoffset+xk-hpsize+psize,
                                        y2-yk-yoffset);
                            g.drawLine(x1+xoffset+xk,
                                        y2-yk-hpsize-yoffset,
                                        x1+xoffset+xk,
                                        y2-yk-hpsize-yoffset+psize);
                        }
                        break;

                    case 11:                 // + symbol plot + square
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            g.drawRect(x1+xoffset+xk-hpsize,
                                        y2-yk-hpsize-yoffset,
                                        psize,psize);
                            g.drawLine(x1+xoffset+xk-hpsize,
                                        y2-yk-yoffset,
                                        x1+xoffset+xk-hpsize+psize,
                                        y2-yk-yoffset);
                            g.drawLine(x1+xoffset+xk,
                                        y2-yk-hpsize-yoffset,
                                        x1+xoffset+xk,
                                        y2-yk-hpsize-yoffset+psize);
                        }
                        break;

                    case 12:                 // Open diamonds
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            g.drawLine(x1+xoffset+xk,
                                        y2-yk-yoffset-hpsize,
                                        x1+xoffset+xk+hpsize,
                                        y2-yk-yoffset);
                            g.drawLine(x1+xoffset+xk+hpsize,
                                        y2-yk-yoffset,
                                        x1+xoffset+xk,
                                        y2-yk+hpsize-yoffset);
                            g.drawLine(x1+xoffset+xk,
                                        y2-yk-yoffset-hpsize,
                                        x1+xoffset+xk-hpsize,
                                        y2-yk-yoffset);
                            g.drawLine(x1+xoffset+xk-hpsize,
                                        y2-yk-yoffset,
                                        x1+xoffset+xk,
                                        y2-yk+hpsize-yoffset);
                        }
                        break;

                    case 13:                 // Open down triangles
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            g.drawLine(x1+xoffset+xk-hpsize,
                                        y2-yk-yoffset-hpsize,
                                        x1+xoffset+xk+hpsize,
                                        y2-yk-yoffset-hpsize);
                            g.drawLine(x1+xoffset+xk-hpsize,
                                        y2-yk-yoffset-hpsize,
                                        x1+xoffset+xk,
                                        y2-yk-yoffset+hpsize);
                            g.drawLine(x1+xoffset+xk+hpsize,
                                        y2-yk-yoffset-hpsize,
                                        x1+xoffset+xk,
                                        y2-yk-yoffset+hpsize);
                        }
                        break;

                    case 14:                 // Open up triangles
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                            g.drawLine(x1+xoffset+xk,
                                        y2-yk-yoffset-hpsize,
                                        x1+xoffset+xk-hpsize,
                                        y2-yk-yoffset+hpsize);
                            g.drawLine(x1+xoffset+xk-hpsize,
                                        y2-yk-yoffset+hpsize,
                                        x1+xoffset+xk+hpsize,
                                        y2-yk-yoffset+hpsize);
                            g.drawLine(x1+xoffset+xk,
                                        y2-yk-yoffset-hpsize,
                                        x1+xoffset+xk+hpsize,
                                        y2-yk-yoffset+hpsize);
                        }
                        break;

                    case 15:  //  Horizontal open oval (use 4 or greater dotSize)
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                                                            maxp = Math.max(psize, 4);
                                                            maxhp = maxp/2;
                            g.drawOval(x1+xoffset+xk-maxhp,
                                        y2-yk-maxhp/2-yoffset,
                                        maxp,maxp/2);
                        }
                        break;

                    case 16:  //  Horizontal filled oval (use 4 or greater dotSize)
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                                                            maxp = Math.max(psize, 4);
                                                            maxhp = maxp/2;
                            g.fillOval(x1+xoffset+xk-maxhp,
                                        y2-yk-maxhp/2-yoffset,
                                        maxp,maxp/2);
                        }
                        break;

                    case 17:  //  vertical open oval (use 5 or greater dotSize)
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                                                            maxp = Math.max(psize, 4);
                                                            maxhp = maxp/2;
                            g.drawOval(x1+xoffset+xk-maxhp/2,
                                        y2-yk-maxhp-yoffset,
                                        maxp/2,maxp);
                        }
                        break;

                    case 18:  //  vertical filled oval (use 5 or greater dotSize)
                        for(k=0; k<npoints[i]; k++){
                            // Don't plot if out of bounds
                            if(x[i][k] < xminex  ||
                                x[i][k] > xmaxex  ||
                                y[i][k] < yminex  ||
                                y[i][k] > ymaxex){continue;}
                            xk=(int)((x[i][k]-xmin)*xscale);
                            yk=(int)((y[i][k]-ymin)*yscale);
                                                            maxp = Math.max(psize, 4);
                                                            maxhp = maxp/2;
                            g.fillOval(x1+xoffset+xk-maxhp/2,
                                        y2-yk-maxhp-yoffset,
                                        maxp/2,maxp);
                        }
                        break;
                    }
                }
            }


            // draw plot legend with box offset by xlegoff and ylegoff
            // from the upper left corner of plot box (which is offset
            // itself by x1 and x2 from the upper left applet corner).
            // Only the curves actually plotted (for which doplot[i] = 1)
            // are included in the legend.

            if(!showLegend){return;}  // Display legend only if showLegend is true

            g.setFont(smallsmallFont);

            // find width of longest legend string & add 25.

            int howMany = 0;    // Number of curves for which doplot = 1
            int widleg=smallsmallFontMetrics.stringWidth(curvetitle[0]);

            for (i=0; i<imax; i++){
                if (doplot[i] == 1) {
                    int widtest=smallsmallFontMetrics.stringWidth(curvetitle[i]);
                    if(widtest > widleg){
                        widleg=widtest;
                    }
                    howMany ++;
                }
            }
            
            if(!StochasticElements.longFormat && !StochasticElements.amPlottingRates) howMany = 
                Math.min(StochasticElements.legendsShortFormat, StochasticElements.numberCurvesToShow) ;
            
            if(!StochasticElements.plotEnergy) howMany--;   // Remove line if no E/dE
            
            if(StochasticElements.longFormat){
                widleg+=25;                                            // Don't add all of this if plotting dE/dt rather than E?
                ylegoff +=5;
            } else {
                widleg += 27;
                xlegoff -= 7;
                ylegoff +=12;
            }

            // Almost double width if two columns (more than 50 curves)

            if(StochasticElements.numberCurvesToShow > 50 
                && StochasticElements.longFormat) widleg = 2*widleg+10;

            int bls=smallsmallFontMetrics.getAscent()+ 2;
            int hgtleg=(Math.min(howMany,50)+1)*bls;
            if(howMany < 50 && StochasticElements.longFormat) hgtleg -= bls;
            g.setColor(dropShadow);
            g.fillRect(x1+xlegoff+2,           // drop shadow
                y1+ylegoff+2,
                widleg+5,
                hgtleg+5);
            g.setColor(legendfg);
            g.fillRect(x1+xlegoff,               // legend box fill
                y1+ylegoff,
                widleg+5,
                hgtleg+5);
            g.setColor(legendbg);
            g.drawRect(x1+xlegoff,           // legend box outline
                y1+ylegoff,
                widleg+5,
                hgtleg+5);

            if(StochasticElements.plotEnergy){
            
                // Write footnote to legends table giving scaling factor for energy curve    
                g.setColor(eplusColor);
                g.setFont(smallsmallFont);
                g.drawString("*Scaled "+decimalPlace(3,AbGraphicsPad.divFac),x1+xlegoff,
                    y1+ylegoff+hgtleg+17);
            }

            howMany = 0;
            int s=1;
            int xlegoffZero = xlegoff;

            // Allow plotting more than 100 curves, but restrict legends to no more than 100 if in tall
            // format and StochasticElements.curvesShortFormat if in short format

            int legMax = Math.min(imax,102);
            if(!StochasticElements.longFormat && !StochasticElements.amPlottingRates) legMax=
                Math.min(StochasticElements.legendsShortFormat,StochasticElements.numberCurvesToShow);
            
            for(int ii=0; ii <legMax; ii++){          // legends

                // Following logic replaces the last legend entry with the energy entry if there are
                // more than 101 isotopes to plot (StochasticElements.curvesShortFormat curves if in short
                // format) but only the largest 101 (StochasticElements.curvesShortFormat curves if in short
                // format) will have legend entries.

                if(ii == 101){
                    i = imax-1;
                } else {
                    i=ii;
                }

                // Set for either 1 column (<= 50 curves) or two columns (> 50 curves)

                if(i<=50){
                    xlegoff = xlegoffZero;
                }else{
                    xlegoff = xlegoffZero + widleg/2 - 3;
                }
                howMany = howMany%51;

                if( doplot[i] == 0 ) { continue; }  // only for active curves
                g.setColor(lcolor[i]);
                g.drawString(curvetitle[i],x1+xlegoff+25, y1+12+ylegoff+bls*howMany);

                switch(mode[i]){

                    case 0:
                        // Draw nothing
                    break;
    
                    case 1:
                        g.drawLine(x1+xlegoff+5,y1+ylegoff +12 - bls/2 + bls*howMany + s,
                            x1+xlegoff+21,y1+12+ylegoff - bls/2 + bls*howMany + s);
                    break;
    
                    case 2:
                        g.drawLine(x1+xlegoff+5,y1+ylegoff +12- bls/2 + bls*howMany + s,
                            x1+xlegoff+21,y1+ylegoff +12 - bls/2 + bls*howMany + s);
                        g.fillOval(x1+xlegoff+13-hpsize,
                            y1+12+ylegoff-bls/2+bls*howMany-hpsize+s,psize,psize);
                    break;
    
                    case 3:
                        g.drawLine(x1+xlegoff+6,y1+ylegoff +12 - bls/2 +bls*howMany + s,
                            x1+xlegoff+10,y1+12+ylegoff - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+16,y1+ylegoff +12 - bls/2 +bls*howMany + s,
                            x1+xlegoff+20,y1+12+ylegoff - bls/2 + bls*howMany + s);
                    break;
    
                    case 4:
                        g.drawOval(x1+xlegoff+7,
                            y1+ylegoff+12-bls/2+bls*howMany+s-hpsize,psize,psize);
                        g.drawOval(x1+xlegoff+17,
                            y1+ylegoff+12-bls/2+bls*howMany+s-hpsize,psize,psize);
                    break;
    
                    case 5:
                        g.fillOval(x1+xlegoff+7,
                            y1+ylegoff+12-bls/2+bls*howMany+s-hpsize,psize,psize);
                        g.fillOval(x1+xlegoff+17,
                            y1+ylegoff+12-bls/2+bls*howMany+s-hpsize,psize,psize);
                    break;
    
                    case 6:
                        g.fillRect(x1+xlegoff+7,
                            y1+ylegoff+12-bls/2+bls*howMany+s-hpsize,psize,psize);
                        g.fillRect(x1+xlegoff+17,
                            y1+ylegoff+12-bls/2+bls*howMany+s-hpsize,psize,psize);
                    break;
    
                    case 7:
                        g.drawRect(x1+xlegoff+7,
                            y1+ylegoff+12-bls/2+bls*howMany-hpsize+s,psize,psize);
                        g.drawRect(x1+xlegoff+17,
                            y1+ylegoff+12-bls/2+bls*howMany-hpsize+s,psize,psize);
                    break;
    
                    case 8:
                        g.drawLine(x1+xlegoff+7-hpsize,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7-hpsize+psize,
                            y1+12+ylegoff -hpsize+psize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+7-hpsize,
                            y1+ylegoff +12 -hpsize +psize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7-hpsize+psize,
                            y1+12+ylegoff -hpsize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17-hpsize,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17-hpsize+psize,
                            y1+12+ylegoff -hpsize+psize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17-hpsize,
                            y1+ylegoff +12 -hpsize +psize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17-hpsize+psize,
                            y1+12+ylegoff -hpsize - bls/2 + bls*howMany + s);
    
                    break;
    
                    case 9:
    
                        g.drawRect(x1+xlegoff+7-hpsize,
                            y1+ylegoff+12 -hpsize - bls/2 + bls*howMany + s,
                            psize, psize);
                        g.drawRect(x1+xlegoff+17-hpsize,
                            y1+ylegoff+12 -hpsize - bls/2 + bls*howMany + s,
                            psize, psize);
                        g.drawLine(x1+xlegoff+7-hpsize,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7-hpsize+psize,
                            y1+12+ylegoff -hpsize+psize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+7-hpsize,
                            y1+ylegoff +12 -hpsize +psize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7-hpsize+psize,
                            y1+12+ylegoff -hpsize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17-hpsize,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17-hpsize+psize,
                            y1+12+ylegoff -hpsize+psize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17-hpsize,
                            y1+ylegoff +12 -hpsize +psize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17-hpsize+psize,
                            y1+12+ylegoff -hpsize - bls/2 + bls*howMany +s);
    
                    break;
    
                    case 10:
    
                        g.drawLine(x1+xlegoff+7-hpsize,
                            y1+ylegoff +12 - bls/2 +bls*howMany + s,
                            x1+xlegoff+7-hpsize+psize,
                            y1+12+ylegoff - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+7,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7,
                            y1+12+ylegoff -hpsize +psize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17-hpsize,
                            y1+ylegoff +12 - bls/2 +bls*howMany + s,
                            x1+xlegoff+17-hpsize+psize,
                            y1+12+ylegoff - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17,
                            y1+12+ylegoff -hpsize +psize - bls/2 + bls*howMany + s);
    
                    break;
    
                    case 11:
    
                        g.drawRect(x1+xlegoff+7-hpsize,
                            y1+ylegoff+12 -hpsize - bls/2 + bls*howMany + s,
                            psize, psize);
                        g.drawRect(x1+xlegoff+17-hpsize,
                            y1+ylegoff+12 -hpsize - bls/2 + bls*howMany + s,
                            psize, psize);
                        g.drawLine(x1+xlegoff+7-hpsize,
                            y1+ylegoff +12 - bls/2 +bls*howMany + s,
                            x1+xlegoff+7-hpsize+psize,
                            y1+12+ylegoff - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+7,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7,
                            y1+12+ylegoff -hpsize +psize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17-hpsize,
                            y1+ylegoff +12 - bls/2 +bls*howMany + s,
                            x1+xlegoff+17-hpsize+psize,
                            y1+12+ylegoff - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17,
                            y1+12+ylegoff -hpsize +psize - bls/2 + bls*howMany + s);
    
                    break;
    
                    case 12:
    
                        g.drawLine(x1+xlegoff+7,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7+hpsize,
                            y1+12+ylegoff - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+7+hpsize,
                            y1+ylegoff +12 - bls/2 +bls*howMany + s,
                            x1+xlegoff+7,
                            y1+12+ylegoff +hpsize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+7,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7-hpsize,
                            y1+12+ylegoff - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+7-hpsize,
                            y1+ylegoff +12 - bls/2 +bls*howMany + s,
                            x1+xlegoff+7,
                            y1+12+ylegoff +hpsize - bls/2 + bls*howMany + s);
    
                        g.drawLine(x1+xlegoff+17,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17+hpsize,
                            y1+12+ylegoff - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17+hpsize,
                            y1+ylegoff +12 - bls/2 +bls*howMany + s,
                            x1+xlegoff+17,
                            y1+12+ylegoff +hpsize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17-hpsize,
                            y1+12+ylegoff - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17-hpsize,
                            y1+ylegoff +12 - bls/2 +bls*howMany + s,
                            x1+xlegoff+17,
                            y1+12+ylegoff +hpsize - bls/2 + bls*howMany + s);
    
                    break;
    
                    case 13:
    
                        g.drawLine(x1+xlegoff+7-hpsize,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7+hpsize,
                            y1+12+ylegoff -hpsize -bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+7-hpsize,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7,
                            y1+12+ylegoff +hpsize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+7 +hpsize,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7,
                            y1+12+ylegoff +hpsize - bls/2 + bls*howMany + s);
    
                        g.drawLine(x1+xlegoff+17-hpsize,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17+hpsize,
                            y1+12+ylegoff -hpsize -bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17-hpsize,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17,
                            y1+12+ylegoff +hpsize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17 +hpsize,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17,
                            y1+12+ylegoff +hpsize - bls/2 + bls*howMany + s);
    
                    break;
    
                    case 14:
    
                        g.drawLine(x1+xlegoff+7,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7-hpsize,
                            y1+12+ylegoff +hpsize -bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+7-hpsize,
                            y1+ylegoff +12 +hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7+hpsize,
                            y1+12+ylegoff +hpsize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+7,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7+hpsize,
                            y1+12+ylegoff +hpsize - bls/2 + bls*howMany + s);
    
                        g.drawLine(x1+xlegoff+17,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17-hpsize,
                            y1+12+ylegoff +hpsize -bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17-hpsize,
                            y1+ylegoff +12 +hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17+hpsize,
                            y1+12+ylegoff +hpsize - bls/2 + bls*howMany + s);
                        g.drawLine(x1+xlegoff+17,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17+hpsize,
                            y1+12+ylegoff +hpsize - bls/2 + bls*howMany + s);
    
                    break;
    
                    case 15:
                        maxp = Math.max(psize, 4);
                        maxhp = maxp/2;
                        g.drawOval(x1+xlegoff+7-maxhp,
                            y1+ylegoff+12-bls/2+bls*howMany+s-maxhp/2,
                            maxp,maxp/2);
    
                        g.drawOval(x1+xlegoff+17-maxhp,
                            y1+ylegoff+12-bls/2+bls*howMany+s-maxhp/2,
                            maxp,maxp/2);
    
                    break;
    
                    case 16:
    
                        maxp = Math.max(psize, 4);
                        maxhp = maxp/2;
                        g.fillOval(x1+xlegoff+7-maxhp,
                            y1+ylegoff+12-bls/2+bls*howMany+s-maxhp/2,
                            maxp,maxp/2);
    
                        g.fillOval(x1+xlegoff+17-maxhp,
                            y1+ylegoff+12-bls/2+bls*howMany+s-maxhp/2,
                            maxp,maxp/2);

                    break;
    
                    case 17:
    
                        maxp = Math.max(psize, 4);
                        maxhp = maxp/2;
    
                        g.drawOval(x1+xlegoff+7-maxhp/2,
                            y1+ylegoff+12-bls/2+bls*howMany+s-maxhp,
                            maxp/2,maxp);
    
                        g.drawOval(x1+xlegoff+17-maxhp/2,
                            y1+ylegoff+12-bls/2+bls*howMany+s-maxhp,
                            maxp/2,maxp);
    
                    break;
    
                    case 18:
    
                        maxp = Math.max(psize, 4);
                        maxhp = maxp/2;
    
                        g.fillOval(x1+xlegoff+7-maxhp/2,
                            y1+ylegoff+12-bls/2+bls*howMany+s-maxhp,
                            maxp/2,maxp);
    
                        g.fillOval(x1+xlegoff+17-maxhp/2,
                            y1+ylegoff+12-bls/2+bls*howMany+s-maxhp,
                            maxp/2,maxp);
    
                    break;

                }
                howMany++;
            }
            
            // Add energy curve if plot is short format and plotEnergy = true
            
            if(StochasticElements.plotEnergy && ! StochasticElements.longFormat){
                howMany = legMax;
                
                g.setColor(lcolor[StochasticElements.numberCurvesToShow]);
                g.drawLine(x1+xlegoff+7-hpsize,
                            y1+ylegoff +12 - bls/2 +bls*howMany + s,
                            x1+xlegoff+7-hpsize+psize,
                            y1+12+ylegoff - bls/2 + bls*howMany + s);
                g.drawLine(x1+xlegoff+7,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+7,
                            y1+12+ylegoff -hpsize +psize - bls/2 + bls*howMany + s);
                g.drawLine(x1+xlegoff+17-hpsize,
                            y1+ylegoff +12 - bls/2 +bls*howMany + s,
                            x1+xlegoff+17-hpsize+psize,
                            y1+12+ylegoff - bls/2 + bls*howMany + s);
                g.drawLine(x1+xlegoff+17,
                            y1+ylegoff +12 -hpsize - bls/2 +bls*howMany + s,
                            x1+xlegoff+17,
                            y1+12+ylegoff -hpsize +psize - bls/2 + bls*howMany + s);

                g.drawString(curvetitle[StochasticElements.numberCurvesToShow],
                            x1+xlegoff+25, y1+12+ylegoff+bls*howMany);
            }

    }

}   // ----- end class GraphicsGoodies2 -----

