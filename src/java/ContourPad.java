// -------------------------------------------------------------------------------------------------------
//  ContourPad creates a graphics Canvas for the frame on which the
//  plotting method can paint the Segre diagram.  Implement the
//  MouseListener interface because we may want to collect and
//  interpret position of mouse clicks on the canvas.  Implement
//  Runnable because we are going to animate the population in the
//  NZ plane.
// -------------------------------------------------------------------------------------------------------


import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

class ContourPad extends Canvas implements MouseListener, Runnable {

    int xoffset;                 // Offset of NZ plot from left
    int yoffset;                 // Offset of NZ plot from top
    int boxWidth;                // Width for isotope boxes
    int boxHeight;               // Height for isotope boxes
    static int zmax;             // Maximum proton number to plot
    static int nmax;             // Maximum neutron number to plot
    int mouseX = 0;              // x-coordinate of mouse
    int mouseY = 0;              // y-coordinate of mouse
    static int protonNumber=1;   // Proton number (Z) for isotope
    static int neutronNumber=1;  // Neutron number (N) for isotope
    static int width;            // Total width of Segre plot
    static int height;           // Total height of Segre plot
    int xmax;                    // x-coordinate for right side of plot
    int ymax;                    // y-coordinate for bottom of plot
    int maxPlotN = 0;            // Max N currently plotted
    int maxPlotZ = 0;            // Max Z currently plotted
    boolean showIsoLabels;       // Show-NoShow isotope labels
    int ts;                      // Max timestep index
    int t;                       // Timestep index
    boolean animateIt = false;   // Flag controlling whether animation runs
    Thread animator = null;      // Thread that the animation runs on
    int sleepTime = 0;         // Animation thread delay (ms)
    int loopDelayFac = 1;        // Additional delay periods after each loop (ms)
    boolean loopFlag = false;    // Whether to loop animation

    double [] contourRange = new double[ContourFrame.numberContours];
    Color [] contourColor = new Color[ContourFrame.numberContours];

    // Neutron numbers for min & max particle-stable mass for given Z

    int [] minDripN = new int[IsotopePad.minDripN.length];
    int [] maxDripN = new int[IsotopePad.maxDripN.length];

    // Boolean array indicating whether isotope is particle stable
    // (that is, this array defines the drip lines)

    boolean [][] isPStable = new boolean [110][200];

    int [] minZDrip = new int[200];

    Color frameColor = MyColors.gray153;          // Line color in Segre
    Color nonSelectColor = MyColors.blueGray;     // Non-selected squares
    Color isoLabelColor = Color.white;            // Isotope labels

    IsoData id;           // Reaction data window for individual isotopes
    AbundanceData ad;     // Abundance window for individual isotopes
    ShowAbundances sa;  // Window to show abundances and mass fractions for isotope

    Font smallFont = new java.awt.Font("Arial", Font.BOLD, 9);
    FontMetrics smallFontMetrics = getFontMetrics(smallFont);
    Font realSmallFont = new java.awt.Font("Arial", Font.PLAIN, 10);
    FontMetrics realSmallFontMetrics = getFontMetrics(realSmallFont);
    Font tinyFont = new java.awt.Font("Arial", Font.PLAIN, 9);
    FontMetrics tinyFontMetrics = getFontMetrics(tinyFont);
    Font bigFont = new java.awt.Font("Arial", Font.PLAIN, 16);
    FontMetrics bigFontMetrics = getFontMetrics(bigFont);
    Font timeFont = new java.awt.Font("Arial", Font.PLAIN, 12);
    FontMetrics timeFontMetrics = getFontMetrics(timeFont);

    Image image;   // Offscreen buffered image
    Graphics ig;    // Offscreen buffered graphics object assoc. with image

    String timerString;          // String to hold time for timer on canvas display
    int timerx;			 // x coordinate for timer
    int timery;                      // y coordinate for timer

    boolean savingFrameNow = false;

    int movieMode = 0;        // Mode for movie frame output.  If movieMode=0, output only 
                                        // the boxes that have changed since the last timestep, except 
                                        // output first the complete frame with labels but empty boxes.  
                                        // (This will be output as the file movies/frameBackground.ps.)
                                        // If movieMode=1, output only the boxes with non-zero values in 
                                        // a timestep, except output first the complete frame with labels.  
                                        // If movieMode=2, output everything (frames, labels, and boxes)
                                        // at each timestep.  Generally, the size of the output .ps files
                                        // for the frames goes up a lot with increasing values of 
                                        // movieMode, particularly for movieMode=2. In modes 0 and 1,
                                        // the first movie frame (frame1.ps) will have all non-zero boxes
                                        // in the first timestep.


    // Initialization block

    {
        xoffset = 58;                            // Offset of NZ plot from left
        yoffset = 35;                            // Offset of NZ plot from top
        boxWidth = ContourFrame.SMALLBOXSIZE;    // Width for isotope boxes
        boxHeight = ContourFrame.SMALLBOXSIZE;   // Height for isotope boxes
        zmax = IsotopePad.zmax;                  // Maximum proton number to plot
        nmax = IsotopePad.nmax;                  // Maximum neutron number to plot);
        ts = StochasticElements.numdt - 1;     // Final timestep interval
        t = ts;

        for (int i=0; i<minDripN.length; i++) {
            minDripN[i] = SegreFrame.gp.minDripN[i];
        }

        for (int i=0; i<maxDripN.length; i++) {
            maxDripN[i] = SegreFrame.gp.maxDripN[i];
        }

        for (int z=0; z<110; z++) {
            for (int n=0; n<200; n++) {
                isPStable[z][n] = SegreFrame.gp.isPStable[z][n];
            }
        }

        // Restore following for plotting purposes

        isPStable[4][4] = true;
        isPStable[5][4] = true;
        isPStable[0][0] = false;  // Failed attempt to suppess (0,0) square
        for (int n=0; n<=SegreFrame.gp.biggestN; n++) {
            minZDrip[n] = SegreFrame.gp.minZDrip[n];
        }

    }

    byte [][][] isoColorIndex = new byte[zmax+1][nmax+1][ts+1];
    byte [][] currentColorIndex = new byte[zmax+1][nmax+1];
    byte [][] previousColorIndex = new byte[zmax+1][nmax+1];
    
    static double currentTime = StochasticElements.timeNow[StochasticElements.numdt -1];



    // ------------------------------------------------------------------------
    //  Public constructor
    // ------------------------------------------------------------------------

    public ContourPad(double [] contourRange, Color [] contourColor) {

        for (int i=0; i<contourRange.length; i++) {
            this.contourRange[i] = contourRange[i];
        }

        for (int i=0; i<contourColor.length; i++) {
            this.contourColor[i] = contourColor[i];
        }

        // Add MouseListener to listen for mouse clicks
        // anywhere on the canvas

        addMouseListener(this);

        width = boxWidth*(nmax+1);
        height = boxHeight*(zmax+1);
        xmax = xoffset + width;
        ymax = yoffset + height;

        // Set canvas size so that ScrollPane knows whether to
        // put scrollbars on the viewport

        this.setSize(xmax+xoffset,ymax+yoffset);
        setColorIndices();
        timerString = StochasticElements.gg.decimalPlace(6,StochasticElements.timeNow[t]);
    }


    // ----------------------------------------------------------------------------------------------------
    //  Implement a run method, as required by the Runnable interface.
    //  The steps of this method will be implemented while the Thread
    //  animator runs.  In this case, run() implements an animation of
    //  the abundances plot in the NZ plane by called repetitively
    //  boxRepainter, which repaints boxes whose population has changed
    //  to reflect the color for the new population.
    // -----------------------------------------------------------------------------------------------------

    public void run() {

        loopFlag = ShowIsotopes.cd.loopcbox[1].getState();

        while(true) {
            t = -1;
            while(animateIt && t++ < ts) {    
                currentTime = StochasticElements.timeNow[t];        
                timerString = StochasticElements.gg.decimalPlace(6,StochasticElements.timeNow[t]);
                ShowIsotopes.cd.timeField.setText(timerString);
                boxRepainter(t);
                try{Thread.sleep(sleepTime);} catch (InterruptedException e){;}
            }
            if( !loopFlag ) break;
            try{Thread.sleep(loopDelayFac*sleepTime);} catch(InterruptedException e){;}
        }

        animator = null;
        ShowIsotopes.cd.animateButton.setLabel("Animate");
    }



    // ----------------------------------------------------------------
    //  Method to start the animation thread
    // ----------------------------------------------------------------

    public void startThread() {
        animator = new Thread(this);
        animator.start();
    }



    // --------------------------------------------------------------------------------------------------
    //  Method setColorIndices to set the color index corresponding
    //  to the population for each isotope at each timestep.  The color
    //  index determines the color that will be displayed for that
    //  isotope at a particular timestep.  The array isoColorIndex[z][n][t]
    //  holds the color index for isotope z and n at timestep t; the
    //  array currentColorIndex[z][n] holds the color index for z and n
    //  at the current timestep.
    // --------------------------------------------------------------------------------------------------

    public void setColorIndices() {
        for(int z=0; z<=zmax; z++) {
            for( int n=minDripN[z]; n<=Math.min(maxDripN[z],nmax); n++ ) {
                if( !isPStable[z][n] ) continue;
                double tryFac;
                for(int t=0; t<=ts; t++) {
                    for(int k=0; k<contourRange.length; k++) {
                        tryFac = StochasticElements.intPop[z][n][t] ;
                        if(!StochasticElements.plotY) tryFac *= ((double)(z+n));
                        if(tryFac <= contourRange[k]){
                            isoColorIndex[z][n][t] = (byte)k;
                            break;
                        }
                    }
                    if(t==ts) currentColorIndex[z][n] = isoColorIndex[z][n][t];
                }
            }
        }
    }




    // ------------------------------------------------------------------------------------------------------
    //  Method boxRepainter to loop over the NZ plane and repaint any
    //  box whose color index has changed since the last step.  Only these
    //  boxes are repainted at each timestep.  To prevent flicker under
    //  Java 1.1 (original version of this program) it is necessary to
    //  override the update() method below.  One major reason:
    //  repaint(x,y,width,height) could be used to repaint a box, but
    //  repaint(...) is only a suggestion to the window manager to repaint.
    //  Found that the loops were too fast to allow the window manager
    //  to repaint in each iteration, so repaints accumulated and were
    //  either done all at once (causing an entire boundingbox rectangle
    //  area to be repainted with corresponding flicker) or it appeared
    //  that some even got dropped.  Solved by overriding update() so
    //  that it did not redraw entire background (the default, with
    //  corresponding flicker).
    // ------------------------------------------------------------------------------------------------------

    public void boxRepainter(int t) {
        for(int z=0; z<zmax; z++) {
            for(int n=minDripN[z]; n<=Math.min(maxDripN[z], nmax-1); n++) {
            
                // Following ensures that correct population will be displayed at any timestep
                // in the animation if one shift-clicks on a box
                
                currentTime = StochasticElements.timeNow[t]; 
   
                StochasticElements.Y[z][n] = StochasticElements.intPop[z][n][t]/StochasticElements.nT;
                
                // reset color index of box if color has changed since last timestep
                if( currentColorIndex[z][n] != isoColorIndex[z][n][t] ) {
                    currentColorIndex[z][n] = isoColorIndex[z][n][t];
                    drawColorSquare(xoffset + n*boxWidth,
                        yoffset + (zmax-z)*boxHeight,
                        boxWidth, boxHeight,
                        contourColor[currentColorIndex[z][n]],
                        frameColor,ig);
                    update(ig);  // update new square to offscreen buffer
                }
            }
        }
        repaint();       // Copy the updated buffer to the screen
    }



    // ---------------------------------------------------------------------------------------------
    //  Override the default implementation of update(g) in order to
    //  suppress its blanking of the full screen before a paint,
    //  which causes flickering updates under non-Swing components.
    // ---------------------------------------------------------------------------------------------

    public void update(Graphics g) {

        drawMesh(xoffset,yoffset,boxWidth,boxHeight,g);
        paint(g);

    }



    // ----------------------------------------------
    //  Paint method
    // ----------------------------------------------

    public void paint(Graphics g){

        // If offscreen buffer image doesn't exist, create it, and
        // define a graphics object ig that is the graphics context
        // for the offscreen buffer image.  We will do all graphics
        // painting to ig, then copy this buffered image all at once
        // to the screen.  This technique prevents screen flickering
        // under Java 1.1.  Later Swing components in Java 2 implement
        // this "double buffering" inherently.

        if (image == null) {
            image = createImage(getSize().width, getSize().height);
            ig = image.getGraphics();
            drawMesh(xoffset,yoffset,boxWidth,boxHeight,ig);
        }

        // Copy the buffered image (created by writing all graphics
        // to the Graphics object ig, which is the graphics context
        // for Image image) to the screen

        g.drawImage(image,0,0,null);

    }



    // --------------------------------------------------------------------------------
    //  Method drawMesh fills boxes, setting any selected
    //  isotopes and isotopes with abundance to colors
    //  corresponding to their contour level.
    // ---------------------------------------------------------------------------------

    void drawMesh(int xoff, int yoff, int w, int h, Graphics g){

        maxPlotN = maxPlotZ = 0;

        if(savingFrameNow){

            // Put time readout on graphics pad so that it will export
            // with PS files of frames.

            putTimer(xoff,yoff,true,true,g);

            // Put contour abundance scale that will export with PS files

            putContourScale(g);
        }

        // Special case for Z=4,5 and N=4 (no entries in reaction lib)

        g.setColor(frameColor);
        g.drawRect(xoffset+4*boxWidth, yoffset+(zmax-4)*boxHeight, boxWidth,boxHeight);

        for (int z=0; z<=zmax; z++){
            for (int n=minDripN[z]; n<=Math.min(maxDripN[z],nmax); n++){
                if( ! isPStable[z][n] ) {continue;}     // omit dripped

                g.setColor(nonSelectColor);
                g.setColor( contourColor[currentColorIndex[z][n]] );

                g.fillRect(xoffset+n*boxWidth,
                        yoffset+(zmax-z)*boxHeight,
                        boxWidth,boxHeight);
                g.setColor(frameColor);
                g.drawRect(xoffset+n*boxWidth, yoffset+(zmax-z)*boxHeight, boxWidth,boxHeight);
                if(n>maxPlotN){maxPlotN = n;}
                if(z>maxPlotZ){maxPlotZ = z;}
            }
        }

        // Put isotope labels on boxes

        if(showIsoLabels) {
            for (int z=1; z<=zmax; z++){
                for(int n=minDripN[z]; n<=Math.min(maxDripN[z],nmax); n++){
                    if( ! isPStable[z][n] ) {continue;}     // Omit dripped
                    setIsoLabel(z,n,g);
                }
            }

            // Special case for neutron

            g.setFont(realSmallFont);
            g.setColor(isoLabelColor);
            int wid = realSmallFontMetrics.stringWidth("n");
            int xzero = xoffset + boxWidth + boxWidth/2 - wid/2;
            int yzero = yoffset + (zmax+1)*boxHeight - boxHeight/2 + 4;
            g.drawString("n",xzero,yzero);    // neutron
        }

        // Following is to suppress the display of (Z,N)=(0,0) square, which pops up
        // in animation when one goes through t=0.  Haven't located the reason yet,
        // so this just overwrites that position with a black square. (Putting if
        // statements to suppress drawing in loops if N=Z=0 and setting isPStable
        // equal to false for (0,0) doesn't seem to work.)  Suppressed when frames are
        // being output to postscript files since they don't have this problem and we
        // don't want an extraneous black square there.

//         if(!savingFrameNow){
//                 g.setColor(Color.black);
//                 g.drawRect(xoffset-1,yoffset+zmax*boxHeight+1,boxWidth,boxHeight);
//                 g.fillRect(xoffset-1,yoffset+zmax*boxHeight+1,boxWidth,boxHeight);
//         }

        // Labels for vertical axis

        g.setFont(smallFont);
        g.setColor(MyColors.gray220);
        for (int z=0; z<=zmax; z += 2) {
            if( minDripN[zmax-z] > nmax ){continue;}
            String tempS = String.valueOf(zmax-z);
            int ds = minDripN[zmax-z]*boxWidth;         // Inset to drip line
            g.drawString(tempS, xoff -8 + ds -smallFontMetrics.stringWidth(tempS),
                yoff + z*h + boxHeight/2 + smallFontMetrics.getHeight()/2 -3);
        }

        // Labels for horizontal axis

        g.setFont(smallFont);
        g.setColor(MyColors.gray220);
        for (int n=1; n<=maxPlotN; n += 2) {
            String tempS = String.valueOf(n);
            g.drawString(tempS, xoff+n*w+boxWidth/2
                -smallFontMetrics.stringWidth(tempS)/2 + 1, yoff+height + 17 - minZDrip[n]*boxHeight);
        }


        // Put general axis labels

        g.setFont(bigFont);
        g.setColor(MyColors.AIyellow);

        HorString hs = new HorString("Neutrons",
                xoff+width/2,yoff+height+20,3,bigFont,bigFontMetrics,g);

        VertString vs = new VertString("Protons",
                xoff-40,yoff+height/2,-2,bigFont,bigFontMetrics,g);


        // Following repaint causes continuous flickering update
        // if not commented out.  It was necessary in IsotopePad
        // to repaint screen if another window covered this one.
        // Restoring the repaint appears to cure the problem that
        // at enlarged size the window may not repaint if uncovered
        // or deiconized, but at the expense of flaky animation.

        //ContourFrame.sp.setScrollPosition(0,5000);
        //repaint();

    }



    // ----------------------------------------------------------------------------------------------------------
    //  Method nonZeroMesh is adaptation of drawMesh for movie frame output
    //  when movieMode=1 (output only of boxes that are non-zero)
    // ----------------------------------------------------------------------------------------------------------

    void nonZeroMesh(int xoff, int yoff, int w, int h, Graphics g){

        maxPlotN = maxPlotZ = 0;

        // Put time readout on graphics pad so that it will export
        // with PS files of frames.

        putTimer(xoff,yoff,true,true,g);

        for (int z=0; z<=zmax; z++){
            for (int n=minDripN[z]; n<=Math.min(maxDripN[z],nmax); n++){
                if( ! isPStable[z][n] ) {continue;}     // omit dripped
                g.setColor(nonSelectColor);
                if(currentColorIndex[z][n] == 0) continue;
                g.setColor( contourColor[currentColorIndex[z][n]] );
                g.fillRect(xoffset+n*boxWidth,yoffset+(zmax-z)*boxHeight, boxWidth,boxHeight);
                g.setColor(frameColor);
                g.drawRect(xoffset+n*boxWidth,yoffset+(zmax-z)*boxHeight,boxWidth,boxHeight);
                if(n>maxPlotN){maxPlotN = n;}
                if(z>maxPlotZ){maxPlotZ = z;}
            }
        }

        // Put isotope labels on boxes

        if(showIsoLabels) {
            for (int z=1; z<=zmax; z++){
                for(int n=minDripN[z]; n<=Math.min(maxDripN[z],nmax); n++){
                    if( ! isPStable[z][n] ) {continue;}     // Omit dripped
                    setIsoLabel(z,n,g);
                }
            }

            // Special case for neutron

            g.setFont(realSmallFont);
            g.setColor(isoLabelColor);
            int wid = realSmallFontMetrics.stringWidth("n");
            int xzero = xoffset + boxWidth + boxWidth/2 - wid/2;
            int yzero = yoffset + (zmax+1)*boxHeight - boxHeight/2 + 4;
            g.drawString("n",xzero,yzero);    // neutron
        }
    }



    // ----------------------------------------------------------------------------------------------------------------
    //  Method diffMesh is adaptation of drawMesh for movie frame output
    //  when movieMode=0 (output only of boxes that have changed color since
    //  previous timestep).
    // ----------------------------------------------------------------------------------------------------------------

    void diffMesh(int xoff, int yoff, int w, int h, Graphics g){

        maxPlotN = maxPlotZ = 0;

        // Put time readout on graphics pad so that it will export
        // with PS files of frames.

        putTimer(xoff,yoff,true,true,g);

        for (int z=0; z<=zmax; z++){
            for (int n=minDripN[z]; n<=Math.min(maxDripN[z],nmax); n++){
                if( ! isPStable[z][n] ) {continue;}     // omit dripped
                g.setColor(nonSelectColor);
                if(currentColorIndex[z][n] == previousColorIndex[z][n]) continue;
                previousColorIndex[z][n] = currentColorIndex[z][n];
                g.setColor( contourColor[currentColorIndex[z][n]] );
                g.fillRect(xoffset+n*boxWidth,yoffset+(zmax-z)*boxHeight,boxWidth,boxHeight);
                g.setColor(frameColor);
                g.drawRect(xoffset+n*boxWidth,yoffset+(zmax-z)*boxHeight,boxWidth,boxHeight);
                if(n>maxPlotN){maxPlotN = n;}
                if(z>maxPlotZ){maxPlotZ = z;}
            }
        }
    }




    // ------------------------------------------------------------------------------------------------------------
    //  Method bkgMesh is adaptation of drawMesh for movie frame output that
    //  outputs the full display (labels and everything), but with population
    //  of all boxes set to zero.  It is used to output the background frame
    //  for the cases movieMode=0,1.
    // ------------------------------------------------------------------------------------------------------------

    void bkgMesh(int xoff, int yoff, int w, int h, Graphics g){

        maxPlotN = maxPlotZ = 0;

        // Place geometrical alignment reference point at (0,0) for all frames

        g.setColor(Color.black);
        g.drawLine(0,0,1,0);

        // Put contour abundance scale that will export with PS files

        putContourScale(g);

        for (int z=0; z<=zmax; z++){
            for (int n=minDripN[z]; n<=Math.min(maxDripN[z],nmax); n++){
                if( ! isPStable[z][n] ) {continue;}     // omit dripped
                g.setColor(nonSelectColor);
                g.setColor( contourColor[0] );  // Set all boxes to zero population
                g.fillRect(xoffset+n*boxWidth,yoffset+(zmax-z)*boxHeight,boxWidth,boxHeight);
                g.setColor(frameColor);
                g.drawRect(xoffset+n*boxWidth,yoffset+(zmax-z)*boxHeight,boxWidth,boxHeight);
                if(n>maxPlotN){maxPlotN = n;}
                if(z>maxPlotZ){maxPlotZ = z;}
            }
        }

        // Labels for vertical axis

        g.setFont(smallFont);
        g.setColor(MyColors.gray220);
        for (int z=0; z<=zmax; z += 2) {
            if( minDripN[zmax-z] > nmax ){continue;}
            String tempS = String.valueOf(zmax-z);
            int ds = minDripN[zmax-z]*boxWidth;         // Inset to drip line
            g.drawString(tempS, xoff -8 + ds -smallFontMetrics.stringWidth(tempS),
                yoff + z*h + boxHeight/2 + smallFontMetrics.getHeight()/2 -3);
        }

        // Labels for horizontal axis

        g.setFont(smallFont);
        g.setColor(MyColors.gray220);
            for (int n=1; n<=maxPlotN; n += 2) {
            String tempS = String.valueOf(n);
            g.drawString(tempS, xoff+n*w+boxWidth/2 -smallFontMetrics.stringWidth(tempS)/2 + 1,
                yoff+height + 17 - minZDrip[n]*boxHeight);
        }


        // Put general axis labels

        g.setFont(bigFont);
        g.setColor(MyColors.AIyellow);

        HorString hs = new HorString("Neutrons",
                xoff+width/2,yoff+height+20,3,bigFont,bigFontMetrics,g);

        VertString vs = new VertString("Protons",
                xoff-40,yoff+height/2,-2,bigFont,bigFontMetrics,g);

        // Put isotope labels on boxes

        if(showIsoLabels) {
            for (int z=1; z<=zmax; z++){
                for(int n=minDripN[z]; n<=Math.min(maxDripN[z],nmax); n++){
                    if( ! isPStable[z][n] ) {continue;}     // Omit dripped
                    setIsoLabel(z,n,g);
                }
            }

            // Special case for neutron

            g.setFont(realSmallFont);
            g.setColor(isoLabelColor);
            int wid = realSmallFontMetrics.stringWidth("n");
            int xzero = xoffset + boxWidth + boxWidth/2 - wid/2;
            int yzero = yoffset + (zmax+1)*boxHeight - boxHeight/2 + 4;
            g.drawString("n",xzero,yzero);    // neutron
        }
    }



    // ----------------------------------------------------------------------------------
    //  Define the 5 methods of the MouseListener interface.
    //  (Must define all once MouseListener is implemented,
    //  even if not all are used.)  Not presently using mouse
    //  events on this graphics canvas except for code below
    //  to allow populations at the end of the calculation to be
    //  displayed by shift-click on the isotope.
    // ----------------------------------------------------------------------------------

    public void mousePressed(MouseEvent me) {

        if ( ( me.isControlDown() || me.isShiftDown() )   // If ctrl or shift
            &&  ( SegreFrame.includeReaction[1]          // is down on press
                || SegreFrame.includeReaction[2]            // & reaction classes
                || SegreFrame.includeReaction[3]            // have been selected
                || SegreFrame.includeReaction[4]
                || SegreFrame.includeReaction[5]
                || SegreFrame.includeReaction[6]
                || SegreFrame.includeReaction[7]
                || SegreFrame.includeReaction[8] ) ) {

            mouseX = me.getX();
            mouseY = me.getY();
            getNZ(mouseX,mouseY);
            
            // Following code allows the final populations for any isotope to be
            // displayed by shift-clicking on it in the final contour animation plot.
            // The populations displayed are the ones corresponding to the current
            // timestep.
            
            if(me.isShiftDown()) {
                IsotopePad.protonNumber = protonNumber;
                IsotopePad.neutronNumber = neutronNumber;
                sa = new ShowAbundances(320,210," Current Abundances", "");
                sa.show();
            }

            

            if( me.isControlDown() ) {             // If ctrl is down on press

            }

        } else {
    
        }   // end of if-block for mousePressed      
    }
        
    public void mouseEntered(MouseEvent me) {   // Not used 
    }

    public void mouseExited(MouseEvent me) {    // Not used  
    }

    public void mouseClicked(MouseEvent me) {

        // The following commented-out code would implement a
        // window launch on double click of an isotope square.
        // Has been replaced by ctrl-press action above.  Note
        // that the methods listed under java.awt.event.InputEvent
        // (superclass of MouseEvent and KeyEvent) could be
        // useful here.
    
        /* 
    
        int clicks = me.getClickCount();  // Check for double click
        if(clicks > 1){
            mouseX = me.getX();
            mouseY = me.getY();
            getNZ(mouseX,mouseY);
            id = new IsoData(600,450," Isotope Data","text2");
            id.show();
        }
    
        */ 

    }
        
    public void mouseReleased(MouseEvent me) {  // Not used  
        //sa.show();
    }




    // ------------------------------------------------------------------------------------
    //  Method getNZ to determine the N and Z corresponding to
    //  the square clicked
    // ------------------------------------------------------------------------------------

    public void getNZ(int x, int y) {

        // Return immediately if outside bounds of the chart
        
        if(x < xoffset || x > xmax || y < yoffset || y > ymax) {
            return;

        // Otherwise determine the N and Z of the clicked square
        // if between the drip lines
    
        } else {
            double fracY = (double)(y-yoffset)/(double)height;
            int tprotonNumber = zmax - ((int)(fracY * (zmax+1)));
            double fracX = (double)(x-xoffset)/(double)width;
            int tneutronNumber = (int)(fracX * (nmax+1));
            if(tneutronNumber < minDripN[tprotonNumber]
                || tneutronNumber > Math.min(maxDripN[tprotonNumber],nmax)) {
                return;
            } else {
                protonNumber = tprotonNumber;
                neutronNumber = tneutronNumber;
            }
        }
    }



    // -------------------------------------------------------------------------------------------
    //  Method drawColorSquare to draw colored square with outline
    // -------------------------------------------------------------------------------------------

    void drawColorSquare(int x, int y, int delx, int dely,Color bgcolor, Color frameColor, Graphics g) {
        if(protonNumber == 0){return;}
        g.setColor(bgcolor);
        g.fillRect(x,y,delx,dely);
        g.setColor(frameColor);
        g.drawRect(x,y,delx,dely);
    }



// -----------------------------------------------------------------------------------
//  Method setIsoLabel to set labels for individual isotopes
// -----------------------------------------------------------------------------------

    void setIsoLabel(int z, int n, Graphics g) {
        if(protonNumber == 0){return;}
        String tempS = returnSymbol(z);
        String tempS2 = String.valueOf(z+n);
        int wid = realSmallFontMetrics.stringWidth(tempS) + tinyFontMetrics.stringWidth(tempS2);
        int xzero = xoffset+n*boxWidth+boxWidth/2-wid/2;
        int yzero = yoffset+(zmax-z+1)*boxHeight-boxHeight/2+1;
        g.setColor(isoLabelColor);
        g.setFont(tinyFont);
        g.drawString(tempS2,xzero,yzero);        // Symbol
        xzero += tinyFontMetrics.stringWidth(tempS2);
        yzero += 5;
        g.setFont(realSmallFont);
        g.drawString(tempS,xzero,yzero);         // Mass Number
    }



    // --------------------------------------------------------------------------------------
    //  Method returnSymbol to return element symbol given the
    //  proton number.
    // --------------------------------------------------------------------------------------

    static String returnSymbol (int z) {

        String [] symbolString = {"","H","He","Li","Be","B","C","N",
                "O","Fl","Ne","Na","Mg","Al","Si","P","S","Cl",
                "Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co",
                "Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb",
                "Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag",
                "Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La",
                "Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho",
                "Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir",
                "Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr",
                "Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk",
                "Cf","Es","Fm","Md","No","Lr"};

        return symbolString [z];
    }



    // ---------------------------------------------------------------------------------------------------------
    //  The method putContourScale puts a contour scale on the graphics pad
    //  for movie export.
    // ---------------------------------------------------------------------------------------------------------

    public void putContourScale(Graphics g){

        Font titleFont = new Font("Arial",Font.BOLD,11);
        FontMetrics titleFontFontMetrics = getFontMetrics(titleFont);
        Font smallFont = new Font("Arial",Font.PLAIN,10);
        FontMetrics smallFontFontMetrics = getFontMetrics(smallFont);

        int topy = 30;
        int vspacing = 12;
        double maxValue = StochasticElements.maxValue;
        int length = contourRange.length;

        int leftSide1 = 40;
        int leftSide2 = leftSide1 + 60;
        int topSide = 25+topy;
        int boxSize = 10;
        int vshift = smallFontFontMetrics.getHeight()-boxSize/2 -1;

        // Create contour legend

        MyColors mc = new MyColors();
        
        int vleg = 11*vspacing-3;
        double inc = 1/(double)vleg;
        for(int i=0; i<vleg; i++) {
            g.setColor( mc.returnRGB(StochasticElements.popColorInvert, 
                StochasticElements.popColorMap, (double)(vleg-i)*inc ) );
            g.drawLine(leftSide1, i+topy+17, 11+leftSide1, i+topy+17);
        }

        String XYlegend = "Abundance Y";
        if(!StochasticElements.plotY) XYlegend = "Mass Frac X";

        g.setColor(MyColors.gray220);
        g.setFont(titleFont);
        g.drawString(XYlegend, leftSide1, topy+5);
        g.setFont(smallFont);

        String temp = StochasticElements.gg.decimalPlace(4,contourRange[0]/StochasticElements.f);
        g.drawString(temp,20+leftSide1,topSide+(length-1)*vspacing);

        for (int i=0; i<length-1; i++) {
            temp = StochasticElements.gg.decimalPlace(4,contourRange[length-1-i]/StochasticElements.f);
            g.drawString(temp,20+leftSide1,topSide+(i)*vspacing);
        }
    }


    // -----------------------------------------------------------------------------------------------------------
    //  The method putTimer puts a timer on the graphics pad for movie export
    // -----------------------------------------------------------------------------------------------------------

    public void putTimer(int xoff, int yoff, boolean withBox, boolean withBkg, Graphics g){

        // Place geometrical alignment reference point at (0,0) for all frames

        g.setColor(Color.black);
        g.drawLine(0,0,1,0);

        timerx = xoff+90;
        timery = yoff + height/25;
        g.setFont(timeFont);

        // Put timer background square if withBox or withBKG are true

        if(withBkg){
                g.setColor(MyColors.gray51);
                g.fillRect(timerx,timery-15,130,22);
        }
        if(withBox){
                g.setColor(MyColors.gray102);
                g.drawRect(timerx,timery-15,130,22);
        }

        // Place time

        g.setColor(MyColors.gray204);
        g.drawString("Time = "+timerString,timerx+7,timery);
    }


    // ----------------------------------------------------------------------------------------------
    //  The method PSfile generates a postscript file of the full
    //  Segre chart.  It uses the class PSGr1, which was obtained
    //  from http://herzberg.ca.sandia.gov/psgr/, and which is
    //  contained in the directory gov that must be present in
    //  this directory, such that the path to the class file is
    //  gov/sandia/postscript relative to this directory (note
    //  the statement above:  import gov.sandia.postscript.PSGr1;).
    //  PSGr1 is for Java 1.1; there is a class PSGr2 for Java 2.
    //  You should be able to open this .ps file in Illustrator,
    //  ungroup, and edit it.  It does not display directly in my
    //  version of Ghostview, but it will display in Ghostview
    //  after saving from Illustrator as a .eps file.  The name
    //  of the postscript file output is contained in the String
    //  fileName.  Default is a file in current directory, but
    //  fileName can include a path to another directory.  For
    //  example, fileName = "..\file.ps" will write the file
    //  file.ps to the parent directory of the current one on
    //  a Windows file system.  Note that the correct portion of
    //  the graphics pad is output to .ps only if it has not been
    //  resized using the size control buttons at the bottom of
    //  the display.
    // ----------------------------------------------------------------------------------------------

    public void PSfile (String fileName) {

        savingFrameNow = true;
    
        try {
            FileWriter psOut = new FileWriter(fileName);
            Graphics postscript = new PSGr1(psOut);
            drawMesh(this.xoffset,this.yoffset,this.boxWidth,this.boxHeight,postscript);
    
            // The following commented-out statement also
            // generates a postscript file of the Segre
            // chart.  However, because it is all copied
            // at once using drawImage from the offscreen image buffer,
            // it is output as a single object that will not
            // ungroup in Illustrator.  To ungroup, the objects
            // appear to need to be written one at a time to the
            // postscript output stream, as done by the
            // drawMesh command above to the postscript graphics
            // object.
        
            //postscript.drawImage(image,0,0,null);
    
        }
        catch (Exception e) {System.out.println(e);
        }
    
        savingFrameNow = false;
    }


    // -----------------------------------------------------------------------------------
    //  Like PSfile above, but outputs only the boxes that have
    //  non-zero values at each timestep.
    // ------------------------------------------------------------------------------------

    public void PSfileNonZero (String fileName) {

        savingFrameNow = true;

        try {
            FileWriter psOut = new FileWriter(fileName);
            Graphics postscript = new PSGr1(psOut);
            nonZeroMesh(this.xoffset,this.yoffset,this.boxWidth,this.boxHeight,postscript);
        }
        catch (Exception e) {System.out.println(e);
        }

        savingFrameNow = false;
    }


    // -----------------------------------------------------------------------------------------------
    //  Like PSfile above, but outputs the entire canvas with all
    //  boxes set to zero population (with no timestep displayed). 
    //  Used to output a constant background frame if movieMode = 0,1.
    // ------------------------------------------------------------------------------------------------

    public void PSfileBkg (String fileName) {

        savingFrameNow = true;

        try {
            FileWriter psOut = new FileWriter(fileName);
            Graphics postscript = new PSGr1(psOut);
            bkgMesh(this.xoffset,this.yoffset,this.boxWidth,this.boxHeight,postscript);
        }
        catch (Exception e) {System.out.println(e);
        }

        savingFrameNow = false;
    }


    // ----------------------------------------------------------------------------------------------------
    //  Like PSfile above, but outputs only those boxes that have changed
    //  color since the last timestep.  Used if movieMode=0.
    // ----------------------------------------------------------------------------------------------------

    public void PSfileDiff (String fileName) {

        savingFrameNow = true;

        try {
            FileWriter psOut = new FileWriter(fileName);
            Graphics postscript = new PSGr1(psOut);
            diffMesh(this.xoffset,this.yoffset,this.boxWidth,this.boxHeight,postscript);
        }
        catch (Exception e) {System.out.println(e);
        }

        savingFrameNow = false;
    }



    // -----------------------------------------------------------------------------------------------------------
    //  Method saveFrames to save movie frames (.ps format) from animation
    //  sequence of population in the NZ plane.
    // -----------------------------------------------------------------------------------------------------------

    public void saveFrames() {

        // First be sure that a subdirectory of the directory from which ElementMaker
        // is running named 'movies' exists and can be written.  Uses the maketheWarning
        // method defined in the class ContourFrame (accessed through the static instance
        // ShowIsotopes.cd).

        File movieDir = new File("movies");
        if( !movieDir.exists() || !movieDir.isDirectory() ){
            String message = "Required subdirectory 'movies' not found.";
            message += " Create it and try again.";
            ShowIsotopes.cd.makeTheWarning(300,300,200,120,Color.black,
                Color.lightGray, " Warning!", message, false, ShowIsotopes.cd );
            return;
        } 

        if( !movieDir.canWrite() ) {
            String message = "Subdirectory 'movies' exists but can't write to it.";
            message += " Set write permission and try again.";
            ShowIsotopes.cd.makeTheWarning(300,300,200,120,Color.black,
                Color.lightGray, " Warning!", message, false, ShowIsotopes.cd );
            return;
        }

        // Launch a progess meter window to indicate that we are
        // outputting frames and how much progress has been made.

        ProgressMeter pm = new ProgressMeter(300,300,205,120,"",
            "Writing to subdirectory 'movies'", "Getting graphics ...");

        // Initialize previous-color index array

        for(int i=0; i<zmax; i++){
            for(int j=0; j<nmax; j++){
                previousColorIndex[i][j] = 0;
            }
        }

        // Save the movie frames as .ps files to the subdirectory 'movies'

        int t = -1;
        while(t++ < ts) {
            timerString = StochasticElements.gg.decimalPlace(6,StochasticElements.timeNow[t]);
            boxRepainter(t);

            // If first frame, output a background frame for the movie

            if(t==0){
                if(movieMode == 2){
                    PSfile("movies/frameBackground.ps");
                } else {
                    PSfileBkg("movies/frameBackground.ps");
                }
                pm.sets2("frameBackground.ps");
            }

            // Frame output depends on value of movieMode

            if(movieMode==0){
                PSfileDiff("movies/frame" + String.valueOf(t+1) +".ps");
            } else if(movieMode==1){
                PSfileNonZero("movies/frame" + String.valueOf(t) +".ps");
            } else {
                PSfile("movies/frame" + String.valueOf(t) +".ps");
            }
            pm.sets2("frame" + String.valueOf(t) +  ".ps");
                        
        }
        pm.makeQuit();    // close the progress meter
    }

}  /* End class ContourPad */

