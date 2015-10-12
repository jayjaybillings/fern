// ----------------------------------------------------------------------------------------------------------
//  Class RatePlotFrame creates a plot of reaction rates as a function
//  of temperature.  Data arrays are filled from an instance of PlotParams.
// ----------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

class RatePlotFrame extends Frame {

    static RateGraphicsPad gp = new RateGraphicsPad();
    
    Font buttonFont = new java.awt.Font("SanSerif", Font.BOLD, 11);
    FontMetrics buttonFontMetrics = getFontMetrics(buttonFont);
    
    
    // ------------------------------------------------------------
    // Public constructor
    // ------------------------------------------------------------
    
    public RatePlotFrame() {
    
        // Lay out a graphics canvas in the frame and
        // add a close button to the frame
    
        setLayout(new BorderLayout());
        add("Center", gp);
    
        Panel bottom = new Panel();
        bottom.setFont(buttonFont);
        Panel left = new Panel();
        Panel top = new Panel();
        Button printButton = new Button(" Print ");
        Button PSButton = new Button (" Write Postscript File ");
        bottom.add(printButton);
        bottom.add(PSButton);
        add("South",bottom);
        add("West",left);
        add("North", top);
    
        // Add inner class event handler for Print button
    
        printButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
                printThisFrame(55,70,false);
            }
        });
    
        // Add inner class event handler for postscript out button.
        // Launches file dialog window to allow the output
        // postscript filename to be specified.
    
        PSButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
    
                RatePlotFileDialogue fd =
                    new RatePlotFileDialogue(100,100,400,110,Color.black,
                        Color.lightGray,"Choose File Name",
                        "Choose a postscript file name:");
                fd.setResizable(false);
                fd.show();
    
            }
        });
    
    
        this.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
            RatePlotFrame.this.hide();
            RatePlotFrame.this.dispose();
            }
        });
    }


    // ----------------------------------------------------------------------------------------------------------
    //  Static method makePS invoked from MyFileDialogue instance to
    //  write postscript file of plots.  gp is not recognized
    //  directly from that object, but by declaring the instances
    //  of RatePlotFrame and RateGraphicsPad to be static, methods of gp
    //  can be invoked indirectly through this method.
    // ----------------------------------------------------------------------------------------------------------

    static void makePS(String file) {
        gp.outputPS(file);
    }



    // -----------------------------------------------------------------------------------------------------------
    //  Method printThisFrame prints an entire Frame (Java 1.1 API).
    //  To print a Frame or class subclassed from Frame, place this
    //  method in its class description and invoke it to print.
    //  It MUST be in a Frame or subclass of Frame.
    //  It invokes a normal print dialogue,
    //  which permits standard printer setup choices. I have found
    //  that rescaling of the output in the print dialogue works
    //  properly on some printers but not on others.  Also, I tried
    //  printing to a PDF file through PDFwriter but that caused
    //  the program to freeze, requiring a hard reboot to recover.
    //  (You can use the outputPS method to output to a .ps file.)
    //
    //      Variables:
    //      xoff = horizontal offset from upper left corner
    //      yoff = vertical offset from upper left corner
    //      makeBorder = turn rectangular border on or off
    // ------------------------------------------------------------------------------------------------------------

    public void printThisFrame(int xoff, int yoff, boolean makeBorder) {

        java.util.Properties printprefs = new java.util.Properties();
        Toolkit toolkit = this.getToolkit();
        PrintJob job = toolkit.getPrintJob(this,"Java Print",printprefs);
        if (job == null) {return;}
        Graphics g = job.getGraphics();
        g.translate(xoff,yoff);     // Offset from upper left corner
        Dimension size = this.getSize();
        if (makeBorder) {           // Rectangular border
            g.drawRect(-1,-1,size.width+2,size.height+2);
        }
        g.setClip(0,0,size.width,size.height);
        this.printAll(g);
        g.dispose();
        job.end();

    }

}  /* End class RatePlotFrame */




// -------------------------------------------------------------------------------------------------------------
//  Class RateGraphicsPad reate a graphics canvas for the frame on which
//  the plotting method can paint.
// -------------------------------------------------------------------------------------------------------------

class RateGraphicsPad extends Canvas {

    // Define some standard colors for convenience

    Color AIyellow=new Color (255,204,0);
    Color AIorange=new Color(255,153,0);
    Color AIred=new Color(204,51,0);
    Color AIpurple=new Color(153,102,153);
    Color AIblue=new Color(102,153,153);
    Color AIgreen=new Color(153,204,153);
    Color gray51=new Color(51,51,51);
    Color gray102=new Color(102,102,102);
    Color gray153=new Color(153,153,153);
    Color gray204=new Color(204,204,204);
    Color gray245=new Color(245,245,245);
    Color gray250=new Color(252,252,252);

    // Define a color array that will be used to fill the
    // color array for the curves in the plotting routine.

    Color [] col = new Color [45];

    {  // initializer

        col[0] = Color.black;
        col[1] = Color.blue;
        col[2] = Color.red;
        col[3] = Color.magenta;
        col[4] = gray102;
        col[5] = new Color(0,220,0);
        col[6] = AIblue;
        col[7] = AIpurple;
        col[8] = AIorange;
        col[9] = AIgreen;

        col[10] = new Color(51,153,51);
        col[11] = new Color(0,51,102);
        col[12] = new Color(0,153,153);
        col[13] = new Color(0,51,153);
        col[14] = new Color(51,153,153);
        col[15] = new Color(0,153,204);
        col[16] = new Color(51,0,153);
        col[17] = new Color(51,204,153);
        col[18] = new Color(153,153,0);
        col[19] = new Color(153,102,51);
        col[20] = new Color(153,51,0);

        col[21] = gray51;
        col[22] = AIred;
        col[23] = gray153;
        col[24] = new Color(153,153,102);
        col[25] = new Color(102,51,153);
        col[26] = new Color(153,51,204);
        col[27] = new Color(153,153,204);
        col[28] = new Color(102,204,255);
        col[29] = new Color(153,51,255);
        col[30] = new Color(255,102,51);

        col[31] = new Color(204,51,102);
        col[32] = new Color(204,153,102);
        col[33] = new Color(204,204,102);
        col[34] = new Color(255,153,102);
        col[35] = new Color(204,51,153);
        col[36] = new Color(204,153,153);
        col[37] = new Color(255,51,153);
        col[38] = new Color(255,153,153);
        col[39] = new Color(255,204,153);
        col[40] = new Color(204,51,204);

        col[41] = new Color(204,153,204);
        col[42] = new Color(255,51,204);
        col[43] = new Color(204,51,255);
        col[44] = new Color(255,51,255);

    }

    // Instantiate the GraphicsGoodies2 class to access its plotIt method

    GraphicsGoodies2 gg=new GraphicsGoodies2();

    //--- Define some parameters that will be needed for plots ----
    //  (Many of these will have their values set or reset externally)

    int plotmode=2;       // Type of plot: 0=linear, 1=log-lin, 2=log-log

    // Set the coordinates (relative to the graphics container
    // from which the method is being invoked) for the plot
    // window. Coordinates in pixels measured from the upper
    // left corner of the container (e.g., a frame or an applet window).

    int x1=0;              // x coordinate of upper left corner of plot
    int y1=0;              // y coordinate of upper left corner of plot
    int x2=483;           // x coordinate of lower right corner of plot
    int y2=490;           // y coordinate of lower right corner of plot

    int kmax=200;         // Max number of points for each curve. Must
                                 // be as large as largest entry in npoints[].
    int imax=40;           // Max number of separate curves

    int numberCurves;     // Actual number of curves to plot;

    int mode [] = new int[imax];
    //int mode[]={4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,4,5,6,7,8};
                          // Array giving display mode for curve
                          // with 1=solid line, 2=lines+dots, 3=dashed line,
                          // 4=open circles, 5=filled circles, 6=filled squares,
                          // 7=open squares, 8=x, 9=x+square, 10=+, 11=+ and square,
                          // 12=open diamonds, 13=open down triangle,
                          // 14=open up triangle, 15=horizontal open oval,
                          // 16=horizontal filled oval, 17=vertical open oval,
                          // 18=vertical filled oval.
                          // If mode other than line mode, dotSize below sets
                          // the size of the plotting symbol.

    int dotSize = 3;      // If plotting dots or symbols, width of dot
                                // in pixels.  Minimum size is 3 pixels.
                                // Smaller sizes will be reset to 3 pixels.

    // Offset of legend box from upper left corner of plot:

    int xlegoff=340;      // x offset in pixels from upper left of frame
    int ylegoff=5;         // y offset in pixels from upper left of frame

    int xdplace=0;        // Number decimal places for numbers on x axis
    int ydplace=0;        // Number decimal places for numbers on y axis

                             // npoints: vector giving number of data points for curve
                             // i with i=0,1, ... imax.  Entries less than
                             // or equal to kmax.  Rule of thumb: if display
                             // mode is lines (mode[i]=2), npoints[i] should
                             // be ~1/2 the graph width in pixels.  If mode
                             // is dots (e.g., mode[i]=1), npoints[i] should
                             // be ~10-15% of graph width in pixels.  If
                             // display mode is dashed line (mode[i]=7),
                             // npoints[i] should be ~1/2-3/4 graph width.

    int npoints[]=new int[imax];   // Values will be set below

    int doscalex=0;   // 1 -> scale min & max of x by data; 0 -> no autoscale
    int doscaley=0;   // 1 -> scale min & max of y by data; 0 -> no autoscale

    int doplot [] = new int[imax];
    //int doplot[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
                             // Vector controlling whether curve
                             // i=0,1,2,...imax is plotted: plot=1; noplot=0
                             // (Reset to toggle curve visibility)

    // Following overridden by autoscaling if doscalex=1 or doscaley=1

    double xmin=1.0E2;           // Min plot x if doscalex=0
    double xmax=1.0E16;        // Max plot x if doscalex=0
    double ymin=1.0E-6;          // Min plot y if doscaley=0
    double ymax=1.0;              // Max plot y if doscaley=0

    // Set the amount of empty space above, below, right, left
    // of the plotted data as a fraction of the total width of
    // the plot area.  For example, delymax=0.10 leaves 10%
    // empty space above the highest data point

    double delxmin=0.0;       // fraction space left on left
    double delxmax=0.0;      // fraction space left on right
    double delymin=0.0;       // fraction space left below
    double delymax=0.0;      // fraction space left above

    // Set the colors for the lines and symbols for the curves
    // (imax entries)

    Color lcolor[] = new Color[imax];

    { // initializer

        for (int i=0; i<imax; i++) {
            lcolor[i] = col[i];
        }
    }

    // Color lcolor[]={AIorange,AIpurple,AIgreen,AIblue,Color.magenta,
    //                 Color.cyan, AIyellow, AIred, Color.blue, gray204,
    //                 Color.orange, gray102, Color.black, gray153,
    //                 Color.red, gray51, Color.green, Color.yellow,
    //                 Color.pink, Color.white};

    // Set the colors for the plot background, labels, & axes,
    // and legend box

    Color bgcolor=Color.white;        // plot background color
    Color axiscolor=gray51;            // axis color
    Color legendfg=gray250;           // legend box color
    Color framefg=Color.white;       // frame color
    Color dropShadow = gray153;    // legend box dropshadow color
    Color legendbg=gray204;           // legend box frame color
    Color labelcolor = gray51;          // axis label color
    Color ticLabelColor = gray51;     // axis tic label color

  // Set the strings for labeling the axes and for
  // each of the curves in the plot legend

    String xtitle="";                                   // String for x-axis label
    String ytitle="";                                   // String for y-axis label
    String curvetitle[]=new String[imax];    // imax entries

    int logStyle = 1;     // 0 to show number, 1 to show log of number
                                // on axis when plot is logarithmic

    int ytickIntervals = 6;      // Number of intervals between y ticks
    int xtickIntervals = 7;      // Number of intervals between x ticks
    boolean showLegend = true;   // Show legend box (true or false)

    // --------------- End of Parameters for Plot ------------------------


    // Create the data arrays.

    double x[][]=new double[imax][kmax];     // data array for x coordinates
    double y[][]=new double[imax][kmax];     // data array for y coordinates
                                                                // Form: x[i][k] and y[i][k], with
                                                                // i=0,1, ... imax = index of separate curves;
                                                                // k=0,1, ... kmax = index of data points


    // --------------------------------------------------------------------------------------------
    //  The invocation of the GraphicsGoodies2.plotIt(...) method
    //  will generally be in a paint method of say a Panel or
    //  Applet object, as illustrated below.
    // --------------------------------------------------------------------------------------------

    public void paint(Graphics g){

        // invoke the GraphicsGoodies2.plotIt() method with the arguments defined above

        gg.plotIt(plotmode,x1,y1,x2,y2,
            kmax,numberCurves,mode,
            dotSize,xlegoff,ylegoff,xdplace,ydplace,
            npoints,doscalex,doscaley,doplot,xmin,xmax,ymin,ymax,
            delxmin,delxmax,delymin,delymax,
            lcolor,bgcolor,axiscolor,legendfg,framefg,
            dropShadow,legendbg,labelcolor,ticLabelColor,
            xtitle,ytitle,curvetitle,logStyle,ytickIntervals,
            xtickIntervals,showLegend,x,y,g);

    }


    // ----------------------------------------------------------------------------------------
    //  The method outputPS generates a postscript file of the
    //  plots.  It uses the class PSGr1, which was obtained
    //  from http://herzberg.ca.sandia.gov/psgr/, and which is
    //  contained in the directory gov that must be present in
    //  this directory, such that the path to the class file is
    //  gov/sandia/postscript relative to this directory (note
    //  the statement above:  import gov.sandia.postscript.PSGr1;).
    //  PSGr1 is for Java 1.1; there is a class PSGr2 for Java 2.
    //  You should be able to open this .ps file in Illustrator,
    //  ungroup, and edit it.  It does not display directly in my
    //  version of ghostview, but it will display in ghostview
    //  after saving from Illustrator as .eps file.  The name
    //  of the postscript file output is contained in the String
    //  fileName.  Default is a file in current directory, but
    //  fileName can include a path to another directory.  For
    //  example, fileName = "..\file.ps" will write the file
    //  file.ps to the parent directory of the current one on
    //  a Windows file system.
    // -------------------------------------------------------------------------------------------

    public void outputPS (String fileName){

        // Create a FileWriter output stream associated with
        // an output file, and pass that stream to PSGr1 as
        // its argument in order to create a new graphics
        // object g that will write to the postscript file.

        try{
            FileWriter fileOut = new FileWriter(fileName);
            Graphics g = new PSGr1(fileOut);

            // invoke the GraphicsGoodies2.plotIt() method with
            // the arguments defined above, passing the
            // graphics context g to it.

            gg.plotIt(plotmode,x1,y1,x2,y2,
                kmax,numberCurves,mode,
                dotSize,xlegoff,ylegoff,xdplace,ydplace,
                npoints,doscalex,doscaley,doplot,xmin,xmax,ymin,ymax,
                delxmin,delxmax,delymin,delymax,
                lcolor,bgcolor,axiscolor,legendfg,framefg,
                dropShadow,legendbg,labelcolor,ticLabelColor,
                xtitle,ytitle,curvetitle,logStyle,ytickIntervals,
                xtickIntervals,showLegend,x,y,g);

         }
         catch (Exception e) {System.out.println(e);}

    }

}  /* End class RateGraphicsPad */

