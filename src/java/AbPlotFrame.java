// -------------------------------------------------------------------------------------------------------------
//  Class AbPlotFrame creates a frame with embedded plot of the calculated
//  abundances as a function of time.
// -------------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

class AbPlotFrame extends Frame {

    static AbGraphicsPad gp;

    // --------------------------------------
    //  Public constructor
    // --------------------------------------
    
    public AbPlotFrame() {
    
        gp = new AbGraphicsPad();
    
        setLayout(new BorderLayout());
        //gp.putData();
        add("Center", gp);
        
        Label spacer1 = new Label();
        Label spacer2 = new Label();
        Panel bottom = new Panel();
        Panel left = new Panel();
        Panel top = new Panel();
        Button printButton = new Button(" Print ");
        Button PSButton = new Button (" Write Postscript File ");
        add("South",bottom);
        add("West",left);
        add("North", top);
    
    
        // Add inner class event handler for Print button
    
        printButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
                printThisFrame(50,125,false);
            }
        });
    
    
        // Add inner class event handler for postscript out button.
        // Launches file dialog window to allow the output
        // postscript filename to be specified.
    
        PSButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
    
                PlotFileDialogue fd =
                    new PlotFileDialogue(100,100,400,110,Color.black,
                            Color.lightGray,"Choose File Name",
                            "Choose a ps file name:");
                fd.setResizable(false);
                fd.show();
    
            }
        });
    
        this.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
            AbPlotFrame.this.hide();
            AbPlotFrame.this.dispose();
            }
        });
    }



    // ----------------------------------------------------------------------------------
    //  Static method makePS invoked from MyFileDialogue instance to write
    //  postscript file of plots.  gp is not recognized
    //  directly from that object, but by declaring the instances
    //  of AbPlotFrame and AbGraphicsPad to be static, methods of gp
    //  can be invoked indirectly through this method.
    // ----------------------------------------------------------------------------------

    static void makePS(String file) {
        gp.outputPS(file);
    }


    // ---------------------------------------------------------------------------------------------------------------
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
    //
    //  NOTE: There is (in late 2009) a general CUPS printing
    //  problem under Linux using the methods employed here that 
    //  throws a  NullPointerException because the printprefs object
    //  is null.  The fix is to put a line
    //
    //       Option orientation-requested 3
    //
    //  into the print specification for every printer in the file 
    //  /etc/cups/printers.conf, and then restart CUPS (/etc/init.d/cups restart)
    // ----------------------------------------------------------------------------------------------------------------

    public void printThisFrame(int xoff, int yoff, boolean makeBorder) {

        java.util.Properties printprefs = new java.util.Properties();
        Toolkit toolkit = this.getToolkit();
        PrintJob job = toolkit.getPrintJob(this,"Java Print",printprefs);
        if (job == null) {return;}
        Graphics g = job.getGraphics();
        g.translate(xoff,yoff);          // Offset from upper left corner
        Dimension size = this.getSize();
        if (makeBorder) {               // Rectangular border
            g.drawRect(-1,-1,size.width+2,size.height+2);
        }
        g.setClip(0,0,size.width,size.height);
        this.printAll(g);
        g.dispose();
        job.end();

    }

}  /* End class AbPlotFrame */



// -----------------------------------------------------------------------------------------------------
//  Class AbGraphicsPad creates a graphics canvas for the frame on
//  which the plotting method can paint
// -----------------------------------------------------------------------------------------------------

class AbGraphicsPad extends Canvas {

    static double divFac;  // Scaling factor for energy plot to keep onscale
    Color Ecolor = new Color(153,204,153);  // Color for energy curve

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


    // GraphicsGoodies2 class will be used for its plotIt method

    GraphicsGoodies2 gg;

    //--- Define the parameters that will be needed for plot ----

    int plotmode=2;       // Type of plot: 0=linear, 1=log-lin, 2=log-log

    // Set the coordinates (relative to the graphics container
    // from which the method is being invoked) for the plot
    // window. Coordinates in pixels measured from the upper
    // left corner of the container (e.g., a frame or an applet window).

    int x1;                   // x coordinate of upper left corner of plot
    int y1;                   // y coordinate of upper left corner of plot
    int x2;                   // x coordinate of lower right corner of plot
    int y2;                   // y coordinate of lower right corner of plot

    int kmax=StochasticElements.tintMax;         // Max number of points for each curve. Must
                                                                     // be as large as largest entry in npoints[].
                            
    int maxToPlot = Math.min(StochasticElements.maxToPlot,StochasticElements.boxPopuli);

    int imax=maxToPlot+1;                // Max number of separate curves to plot + 1 

    int numberCurves = maxToPlot;    // Actual number of curves to plot
                                                      // Can't be greater than imax-1
    int plotWhat;                                // Plot mass fraction (=0), abundance (=1),
                    
    int mode[] = new int[imax];         // Array giving display mode for curve
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

    int xlegoff;                 // x offset in pixels from upper left of frame
    int ylegoff;                 // y offset in pixels from upper left of frame

    int xdplace=StochasticElements.xdeci;   // Number decimal places for numbers on x axis
    int ydplace=StochasticElements.ydeci;   // Number decimal places for numbers on y axis

    int npoints[]=new int[imax];   // Vector gives number of data points for curve
                                              //  i with i=0,1, ... imax.  Entries less than
                                              // or equal to kmax.  Rule of thumb: if display
                                              // mode is lines (mode[i]=2), npoints[i] should
                                              // be ~1/2 the graph width in pixels.  If mode
                                              // is dots (e.g., mode[i]=1), npoints[i] should
                                              // be ~10-15% of graph width in pixels.  If
                                              // display mode is dashed line (mode[i]=7),
                                              // npoints[i] should be ~1/2-3/4 graph width.
                                              // Values will be set below

    int doscalex=0;     // 1 -> scale min & max of x by data; 0 -> no autoscale
    int doscaley=0;     // 1 -> scale min & max of y by data; 0 -> no autoscale

    int doplot[] = new int[imax];              // Vector controlling whether curve
                                                          // i=0,1,2,...imax is plotted: plot=1; noplot=0
                                                          // (Reset to toggle curve visibility)

    // Following overridden by autoscaling if doscalex=1 or doscaley=1

    double xmin;  // Set below // 1.0; //1.0E2;  // Min plot x if doscalex=0
    double xmax;  // Set below //1.0E3; //1.0E16;// Max plot x if doscalex=0

    double ymin = Math.pow(10, StochasticElements.yminPlot);                 // Min plot y if doscaley=0
    double ymax = Math.pow(10, StochasticElements.ymaxPlot);               // Max plot y if doscaley=0

    boolean linesOnly = StochasticElements.linesOnly;     // Plot only with lines (no symbols)
    boolean blackOnly = StochasticElements.blackOnly ;  // Plot only in black (no colors)

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
            if(blackOnly){
                    lcolor[i] = Color.black;
            } else {
                    lcolor[i] = col[i%43];
            }
            if(linesOnly){
                    mode[i] = 1;
            } else {
                    mode[i] = (i%17)+1;
                    if(mode[i] == 3)mode[i] = 18;     // override dashed lines
            }
            doplot[i] = 1;
        }
    }


    // Set the colors for the plot background, labels, & axes,
    // and legend box

    Color bgcolor=Color.white;         // plot background color
    Color axiscolor=gray51;             // axis color
    Color legendfg=gray250;            // legend box color
    Color framefg=Color.white;        // frame color
    Color dropShadow = gray153;    // legend box dropshadow color
    Color legendbg=gray204;           // legend box frame color
    Color labelcolor = gray51;          // axis label color
    Color ticLabelColor = gray51;     // axis tic label color

    // Set the strings for labeling the axes and for
    // each of the curves in the plot legend

    String xtitle="Time (seconds)";             // String for x-axis label
    String ytitle="Mass Fraction";                // y-axis label (may be overridden below)
    String curvetitle[]=new String[imax];    // imax entries

    int logStyle = 1;     // 0 to show number, 1 to show log of number
                                // on axis when plot is logarithmic

    int ytickIntervals = StochasticElements.ytics;    // Number of intervals between y ticks
    int xtickIntervals = StochasticElements.xtics;    // Number of intervals between x ticks
    boolean showLegend = true;                             // Show legend box (true or false)

    // --------------------- End of Parameters for Plot ------------------------------

        
    int tempdim = StochasticElements.pmax*StochasticElements.nmax;
    int [] plotVectorZ = new int [tempdim];               // Holds Zs for plotted isotopes
    int [] plotVectorN = new int [tempdim];               // Holds Ns for plotted isotopes
    double [] plotMaxVal = new double [tempdim];    // Max mass fraction for isotope

    double plotMax;                       // Max value to be plotted
    double plotMin;                        // Min value to be plotted

    // Create the data arrays. They will be filled in the putData() method defined below

    // For the first plot
    double x[][]=new double[imax][kmax];    // data array for x coordinates
    double y[][]=new double[imax][kmax];    // data array for y coordinates
                                                               // Form: x[i][k] and y[i][k], with
                                                               // i=0,1, ... imax = index of separate curves;
                                                               // k=0,1, ... kmax = index of data points

    
    // ------------------------------------------------ Constructor --------------------------------------------------------------

    public AbGraphicsPad() {
    
        // Set the coordinates (relative to the graphics container
        // from which the method is being invoked) for the plot
        // window. Coordinates in pixels measured from the upper
        // left corner of the container (e.g., a frame or an applet window).
    
        if(StochasticElements.longFormat){
            x1= -38;               // x coordinate of upper left corner of plot
            y1=0;                   // y coordinate of upper left corner of plot
            x2= x1+440;         // x coordinate of lower right corner of plot
            y2=565;               // y coordinate of lower right corner of plot
            
            // Offset of legend box from upper left corner of plot:

            xlegoff=x2+35;          // x offset in pixels from upper left of frame
            ylegoff=0;                 // y offset in pixels from upper left of frame
            
        } else {
            
            // Short vertical format for line plot
            x1= -38;               // x coordinate of upper left corner of plot
            y1=0;                   // y coordinate of upper left corner of plot
            x2= x1+512;         // x coordinate of lower right corner of plot
            y2=385;               // y coordinate of lower right corner of plot
            
            // Offset of legend box from upper left corner of plot:

            xlegoff=x2+30;          // x offset in pixels from upper left of frame
            ylegoff=0;                 // y offset in pixels from upper left of frame
        }
        
        // Instantiate the GraphicsGoodies2 class to access its plotIt method
        
        gg=new GraphicsGoodies2();
        
        // Put data in the graphics arrays
        
        putData();

    }


    // --------------------------------------------------------------------------------------------------------
    //  Method to determine which curves to plot.  It orders all curves
    //  in descending order of maximum mass fraction or abundance within
    //  the time range to be plotted.
    // --------------------------------------------------------------------------------------------------------

    void whichCurves() {

        int zmax = StochasticElements.pmax;
        int nmax = StochasticElements.nmax;
        int numdt = StochasticElements.numdt;
        double mf;
        int includeCount = 0;
        double maxThisCurve;

        for(int i=0; i<zmax; i++) {                // loop over Z
            for(int j=0; j<nmax; j++) {            // loop over N
                maxThisCurve = 0;
                for(int k=1; k<numdt; k++) {    // loop over timesteps
                    // Compute mass fraction or abundance for this Z, N, timestep
                    if(plotWhat==1){
                            mf = StochasticElements.intPop[i][j][k]/StochasticElements.f;
                    } else {
                            mf = StochasticElements.intPop[i][j][k]*(i+j)/StochasticElements.f;
                    }

                    if(mf > maxThisCurve) maxThisCurve = mf;
                }
                plotVectorZ[includeCount] = i;
                plotVectorN[includeCount] = j;
                plotMaxVal[includeCount] = maxThisCurve;
                includeCount++;
            }
        }

        // Bubble sort according to max mass fraction in the plot interval

        int nswaps = 1;
        double tempSwap;
        int intSwap;
        while(nswaps > 0){
            nswaps = 0;
            for(int i=0; i<tempdim-1; i++){
                if(plotMaxVal[i+1] > plotMaxVal[i]){

                    tempSwap = plotMaxVal[i];
                    plotMaxVal[i] = plotMaxVal[i+1];
                    plotMaxVal[i+1] = tempSwap;

                    intSwap = plotVectorZ[i];
                    plotVectorZ[i] = plotVectorZ[i+1];
                    plotVectorZ[i+1] = intSwap;

                    intSwap = plotVectorN[i];
                    plotVectorN[i] = plotVectorN[i+1];
                    plotVectorN[i+1] = intSwap;

                    nswaps++;
                }
            }
        }

        // List maximum in time integration interval for all populations

        StochasticElements.toChar.println();
        String tmpstringer = "X";
        if(plotWhat==1){
            StochasticElements.toChar.println("Maximum abundances Y in integration range:");
            tmpstringer = "Y";
        } else {
            StochasticElements.toChar.println("Maximum mass fractions X in integration range:");
        }
        StochasticElements.toChar.println();
        for(int i=0; i<StochasticElements.numberIsotopesPopulated; i++){
            StochasticElements.toChar.println((i+1)+"  Z="+plotVectorZ[i]+" N="+plotVectorN[i]
                    +" Max "+tmpstringer+"="+gg.decimalPlace(4,plotMaxVal[i]) );
        }

        // Determine the min and max values to be plotted

        plotMax = plotMin = StochasticElements.intPop[plotVectorZ[0]][plotVectorN[0]] [0];
        double tryIt;
        int Amax=0;
        int Amin=1000;

        for (int i=0; i<includeCount; i++) {
            for (int j=0; j<StochasticElements.numdt; j++) {
                tryIt = StochasticElements.intPop[plotVectorZ[i]][plotVectorN[i]] [j];
                if(tryIt >= plotMax) {
                    plotMax = tryIt;
                    Amax = plotVectorZ[i] + plotVectorN[i];
                }
                if(tryIt <= plotMin) {
                    plotMin = tryIt;
                    Amin = plotVectorZ[i] + plotVectorN[i];
                }
            }
        }

        switch (plotWhat) {

            case 0:                 // mass fractions

                plotMax = plotMax*Amax/StochasticElements.f;
                plotMin = plotMin*Amin/StochasticElements.f;
                                ytitle="X";
                break;

            case 1:                 // abundances

                plotMax = plotMax/StochasticElements.f;
                plotMin = plotMin/StochasticElements.f;
                                ytitle="Y";
                break;

            case 2:                 // raw counts

                break;
        }
        
        String ytitle2 = " legends 101 largest)";
        if(!StochasticElements.longFormat) 
            ytitle2 = " legends "+StochasticElements.legendsShortFormat+" largest)";

        if(numberCurves > 101 && numberCurves >= StochasticElements.maxToPlot){
            ytitle += (" ("+numberCurves+" largest of "+StochasticElements.numberIsotopesPopulated
                    +";" +ytitle2 );
        }

        if(numberCurves > 101 && numberCurves < StochasticElements.maxToPlot){
            ytitle += (" ("+numberCurves+" isotopes;" + ytitle2 );
        }
    }


    // -------------------------------------------------------
    //  Method to fill data arrays for plotting
    // -------------------------------------------------------

    void putData () {

        // Following holds number plotted curves for GraphicsGoodies to read

        StochasticElements.numberCurvesToShow = numberCurves;  

        if(StochasticElements.plotY){
            plotWhat = 1;
        } else {
            plotWhat = 0;
        }

        // Set xmin and xmax from their values in StochasticElements

        xmin = Math.pow(10, StochasticElements.logtminPlot);
        xmax = Math.pow(10, StochasticElements.logtmaxPlot);

        whichCurves();
        
        boolean showFluxInstead = false;           // If true and Y chosen, flux ratio plotted instead  of Y
                                                                  // If X chosen, normal mass fraction plot no matter what

        double floorVal = ymin/1e8;
    
        for (int i=0; i<numberCurves; i++) {
        
            curvetitle[i] = " Z=" + String.valueOf(plotVectorZ[i]) +" N="  + String.valueOf(plotVectorN[i]);
            npoints[i] = StochasticElements.numdt;

            for (int k=0; k<StochasticElements.numdt; k++) {
                x[i][k] = StochasticElements.timeNow[k];
                int Z = plotVectorZ[i];
                int N = plotVectorN[i];
                if(StochasticElements.intPop[Z][N][k] <= 0) {
                    y[i][k] = floorVal;          // Handle log 0 or log neg
                } else {
                    switch(plotWhat){
                        case 0:
                            y[i][k]=StochasticElements.intPop[Z][N][k]*(Z+N)/StochasticElements.f;
                        break;

                        case 1:
                            if(showFluxInstead){                                                     // Flux ratio instead of Y
                                double deno = StochasticElements.sFplus[Z][N][k] 
                                            +StochasticElements.sFminus[Z][N][k];
                                if(deno > 0){
                                y[i][k]=Math.abs(StochasticElements.sFplus[Z][N][k] 
                                        - StochasticElements.sFminus[Z][N][k]) /deno;
                                } else {
                                    y[i][k] = 1;
                                }
                            } else {
                                y[i][k]=StochasticElements.intPop[Z][N][k]/StochasticElements.f;  // Regular Y
                            }                       
                        break;
                    }
                }
            }
        }


        // Add energy release curve

        double eMax = 0;
        int cindex;
        
        for (int k=0; k<StochasticElements.numdt; k++) {
            x[numberCurves][k] = StochasticElements.timeNow[k];

            // Math.max to prevent attempt to plot log(0) if E = 0.  
            // Absolute value of eNow or deNow to allow log plot if they are negative.
            // Energies are per test particle.  Depending on value of boolean 
            // StochasticElements.plotdE, following displays energy integrated
            // to this time or dE/dt at this time.

            if(StochasticElements.plotEnergy){
            
                if(StochasticElements.plotdE){
                    y[numberCurves][k] = Math.max(Math.abs(StochasticElements.deNow[k]),1.0E-20);  // dE/dt
                } else {
                    y[numberCurves][k] = Math.max(Math.abs(StochasticElements.eNow[k]),1.0E-20);  // Total E
                }            
                if ( y[numberCurves][k]>eMax ) {eMax=y[numberCurves][k];}
            }
        }


        npoints[numberCurves] = StochasticElements.numdt;
        mode[numberCurves] = 10;            // Solid line = 1, open circle=4, closed circle=5, + = 10
        lcolor[numberCurves] = Ecolor;      // Energy color

        // Following scales the energy output curve by a power of 10
        // so that the max value lies between 0.1 and 1 for plotting convenience

        double eScaler = 0.434448229*Math.log(eMax);
        int ieScaler = Math.round((float)eScaler + 0.5F);
        divFac = Math.pow(10,ieScaler);
        for (int k=0; k<StochasticElements.numdt; k++) {
            y[numberCurves][k] /= divFac;
        }

        if(StochasticElements.plotEnergy){
            if(StochasticElements.plotdE){
                curvetitle[numberCurves] = " |dE/dt|*";
            } else {
                curvetitle[numberCurves] = " |sum E|*";
            }
        } else {
            curvetitle[numberCurves] = ""; 
            mode[numberCurves] = 0;

            // Note: changing above line to numberCurves
        }
        
    }


    // ------------------------------------------------------------------------------------------
    //  The invocation of the GraphicsGoodies2.plotIt(...) method
    //  will generally be in a paint method of say a Panel or
    //  Applet object, as illustrated below.
    // ------------------------------------------------------------------------------------------

    public void paint(Graphics g){

        // invoke the GraphicsGoodies2.plotIt() method with
        // the arguments defined above

        gg.plotIt(plotmode,x1,y1,x2,y2,
            StochasticElements.numdt,numberCurves+1,mode,
            dotSize,xlegoff,ylegoff,xdplace,ydplace,
            npoints,doscalex,doscaley,doplot,xmin,xmax,ymin,ymax,
            delxmin,delxmax,delymin,delymax,
            lcolor,bgcolor,axiscolor,legendfg,framefg,
            dropShadow,legendbg,labelcolor,ticLabelColor,
            xtitle,ytitle,curvetitle,logStyle,ytickIntervals,
            xtickIntervals,showLegend,x,y,g);
    }


    // -------------------------------------------------------------------------------------------
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
    // --------------------------------------------------------------------------------------------


    public void outputPS (String fileName){

        // Create a FileWriter output stream associated with
        // an output file, and pass that stream to PSGr1 as
        // its argument in order to create a new graphics
        // object g that will write to the postscript file.

        try {
            FileWriter fileOut = new FileWriter(fileName);
            Graphics g = new PSGr1(fileOut);

            // invoke the GraphicsGoodies2.plotIt() method with
            // the arguments defined above, passing the
            // graphics context g to it.

            gg.plotIt(plotmode,x1,y1,x2,y2,
                StochasticElements.numdt,numberCurves+1,mode,
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

}  /* End class AbGraphicsPad */

