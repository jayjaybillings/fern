// -----------------------------------------------------------------------------------------------------
// IsotopePad creates a graphics Canvas for the frame on which the
// plotting method can paint.  Implement the MouseListener
// interface because we want to collect and interpret
// position of mouse clicks on the canvas in order to
// toggle square colors.
// -----------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

class IsotopePad extends Canvas implements MouseListener {

    static boolean updateAbundance = false;  // used to get square to update
                                                                 // to right color after initial
                                                                 // abundance set

    int xoffset = 58;            // Offset of NZ plot from left
    int yoffset = 35;            // Offset of NZ plot from top
    int boxWidth = SegreFrame.SMALLBOXSIZE;      // Width for isotope boxes
    int boxHeight = SegreFrame.SMALLBOXSIZE;     // Height for isotope boxes
    static int zmax = StochasticElements.pmaxPlot;   // Max proton number to plot
    static int nmax = StochasticElements.nmaxPlot;   // Max neutron number to plot
    int mouseX = 0;                  // x-coordinate of mouse
    int mouseY = 0;                  // y-coordinate of mouse
    static int protonNumber=1;   // Proton number (Z) for isotope
    static int neutronNumber=1;  // Neutron number (N) for isotope
    static int width;                    // Total width of Segre plot
    static int height;                  // Total height of Segre plot
    int xmax;                            // x-coordinate for right side of plot
    int ymax;                            // y-coordinate for bottom of plot
    int maxPlotN = 0;                // Max N currently plotted
    int maxPlotZ = 0;                 // Max Z currently plotted
    int biggestN;
    boolean showIsoLabels;       // Show-NoShow isotope labels
    static boolean[][]isoColor = new boolean[DataHolder.Znum][DataHolder.Nnum];  // Isotope selected
    static boolean[][]isAbundant = new boolean[DataHolder.Znum][DataHolder.Nnum]; // Seed isotopes+H+He

    // Neutron numbers for min mass for given Z in reaclib library

    static int [] minDripN = {
        1,0,1,3,3,3,3,4,5,5,
        5,6,7,8,8,8,8,9,9,10,
        10,11,12,13,14,15,16,17,18,19,
        21,22,23,24,25,26,27,29,30,31,
        32,33,35,36,37,38,40,41,42,43,
        44,46,47,48,49,51,52,53,55,56,
        58,59,61,62,64,65,67,69,70,72,
        73,75,77,78,80,81,83,85,87,88,
        90,92,93,95,98,101};

    // Neutron numbers for max mass for given Z in reaclib libary

    static int [] maxDripN = {
        1,2,4,6,8,9,12,14,14,17,
        31,33,35,38,40,42,44,46,49,51,
        53,55,58,60,62,64,66,69,71,73,
        75,77,80,82,84,86,88,91,93,95,
        97,99,102,104,106,108,110,113,115,117,
        119,121,124,126,128,130,133,134,137,139,
        141,144,146,148,150,153,155,157,159,161,
        164,166,168,170,173,175,177,179,182,184,
        186,187,191,193,192,194};

    // Boolean array indicating whether isotope is particle stable (this array defines the drip lines)

    boolean [][] isPStable = new boolean [110][200];

    int [] minZDrip = new int[200];

    Color selectColor = MyColors.AIpurple;
    Color frameColor = MyColors.gray204;
    Color nonSelectColor = new Color(0,0,180);
    Color isoLabelColor = Color.white;
    Color initAbundColor = new Color(230,185,0);

    IsoData id;                  // Reaction data window for individual isotopes
    AbundanceData ad;     // Abundance window for individual isotopes

    Font smallFont = new java.awt.Font("SanSerif", Font.PLAIN, 9);
    FontMetrics smallFontMetrics = getFontMetrics(smallFont);
    Font realSmallFont = new java.awt.Font("SanSerif", Font.PLAIN, 10);
    FontMetrics realSmallFontMetrics = getFontMetrics(realSmallFont);
    Font tinyFont = new java.awt.Font("SanSerif", Font.PLAIN, 9);
    FontMetrics tinyFontMetrics = getFontMetrics(tinyFont);
    Font bigFont = new java.awt.Font("SanSerif", Font.PLAIN, 16);
    FontMetrics bigFontMetrics = getFontMetrics(bigFont);

    Graphics ig;
    Image image;


    // ---------------------------------------
    //  Public constructor
    // ---------------------------------------

    public IsotopePad(){

        // Add MouseListener to listen for mouse clicks
        // anywhere on the canvas

        addMouseListener(this);

        //this.setBackground(Color.black);

        width = boxWidth*(nmax+1);
        height = boxHeight*(zmax+1);
        xmax = xoffset + width;
        ymax = yoffset + height;

        // Set canvas size so that ScrollPane knows whether to
        // put scrollbars on the viewport

        this.setSize(xmax+xoffset,ymax+yoffset);
        initPStable();       // initialize drip line info
    }


    // ---------------------------------------------------------------------------------
    //  Method to initialize the arrays determining where the
    //  drip lines are and related quantities.
    // ---------------------------------------------------------------------------------

    void initPStable() {

        // Generic fill with "true" if between min and max mass
        // particle stable isotopes

        biggestN = 0;

        for (int z=0; z<= zmax; z++) {
            int highN = Math.min(nmax, maxDripN[z]);
            for (int n=minDripN[z]; n<=highN; n++) {
                isPStable[z][n] = true;
                if(highN > biggestN){biggestN = highN;}
            }
        }

        // Handle the "drips" as special cases.  Following have
        // no entries in Friedel reaction library.

        isPStable[4][4] = false;
        isPStable[5][4] = false;

        // For later plotting convenience, determine array of
        // min particle-stable Z for given N

        for (int n=0; n<=biggestN; n++) {
            for (int z=0; z<=zmax; z++) {
                if ( isPStable[z][n] ) {
                    minZDrip[n] = z;
                    break;
                }
            }
        }
    }



    // ----------------------------------------------------------------------------
    //  Method drawMesh fill boxes, setting any selected
    //  isotopes and isotopes with initial abundance to
    //  special colors.
    // ----------------------------------------------------------------------------

    void drawMesh(int xoff, int yoff, int w, int h, Graphics g){

        maxPlotN = maxPlotZ = 0;

        // Special case for Z=4,5 and N=4 (no entries in reaction lib)

        g.setColor(frameColor);
        g.drawRect(xoffset+4*boxWidth,
            yoffset+(zmax-4)*boxHeight,
            boxWidth,boxHeight);

        for (int z=0; z<=zmax; z++){
            for (int n=minDripN[z]; n<=Math.min(maxDripN[z],nmax); n++){
                if( ! isPStable[z][n] ) {continue;}      // omit dripped
                if( isAbundant[z][n] ) {
                    g.setColor(initAbundColor);
                } else if( isoColor[z][n] ){
                    g.setColor(selectColor);
                } else {
                    g.setColor(nonSelectColor);
                }
                g.fillRect(xoffset+n*boxWidth,
                    yoffset+(zmax-z)*boxHeight,
                    boxWidth,boxHeight);
                g.setColor(frameColor);
                g.drawRect(xoffset+n*boxWidth,
                    yoffset+(zmax-z)*boxHeight,
                    boxWidth,boxHeight);
                if(n>maxPlotN){maxPlotN = n;}
                if(z>maxPlotZ){maxPlotZ = z;}
            }
        }

        // Put isotope labels on boxes

        if(showIsoLabels) {
            g.setColor(isoLabelColor);
            for (int z=1; z<=zmax; z++){
                for(int n=minDripN[z]; n<=Math.min(maxDripN[z],nmax); n++){
                    if( ! isPStable[z][n] ) {continue;}     // Omit dripped
                    setIsoLabel(z,n,g);
                }
            }

            // Special case for neutron

            g.setFont(realSmallFont);
            int wid = realSmallFontMetrics.stringWidth("n");
            int xzero = xoffset + boxWidth + boxWidth/2 - wid/2;
            int yzero = yoffset + (zmax+1)*boxHeight - boxHeight/2 +4;
            g.drawString("n",xzero,yzero);    // neutron
        }

        // Labels for vertical axis

        g.setFont(smallFont);
        g.setColor(Color.white);
        int labelStep = 1;
        for (int z=0; z<=zmax; z+=labelStep) {
        if( minDripN[zmax-z] > nmax ){continue;}
        String tempS = String.valueOf(zmax-z);
        int ds = minDripN[zmax-z]*boxWidth;        // Inset to drip line
        g.drawString(tempS,
            xoff -8 + ds -smallFontMetrics.stringWidth(tempS),
            yoff + z*h + boxHeight/2
            + smallFontMetrics.getHeight()/2 -3);
        }

        // Labels for horizontal axis

        g.setFont(smallFont);
        g.setColor(Color.white);
            labelStep = 1;
            if(maxPlotN > 99 || boxWidth==SegreFrame.SMALLBOXSIZE)labelStep=2;
        for (int n=1; n<=maxPlotN; n+=labelStep) {
        String tempS = String.valueOf(n);
        g.drawString(tempS,
            xoff+n*w+boxWidth/2
            -smallFontMetrics.stringWidth(tempS)/2 + 1,
            yoff+height + 17 - minZDrip[n]*boxHeight);
        }

        // Put general axis labels

        g.setFont(bigFont);
        g.setColor(Color.white);

        HorString hs = new HorString("Neutrons",
            xoff+width/2,yoff+height+25,3,bigFont,bigFontMetrics,g);

        VertString vs = new VertString("Protons",
            xoff-40,yoff+height/2,-2,bigFont,bigFontMetrics,g);

        SegreFrame.sp.setScrollPosition(0,5000);
        repaint();
    }


    // -----------------------------------------------------------------------------------------------------------
    //  Define the 5 methods of the MouseListener interface.
    //  (Must define all once MouseListener is implemented,
    //  even if not all are used.)

    //  Use mousePressed events to select and deselect isotope
    //  squares; use mousePressed + ctrl to launch new window
    //  allowing detailed editing for individual isotopes and
    //  mousePressed + Shift to launch new window allowing
    //  initial abundance to be specified.
    // -----------------------------------------------------------------------------------------------------------

    public void mousePressed(MouseEvent me) {

        // If ctrl or shift is down on press and reaction classes have been selected

        if ( ( me.isControlDown() || me.isShiftDown() ) 
            &&  ( SegreFrame.includeReaction[1]    
            || SegreFrame.includeReaction[2]     
            || SegreFrame.includeReaction[3]   
            || SegreFrame.includeReaction[4]
            || SegreFrame.includeReaction[5]
            || SegreFrame.includeReaction[6]
            || SegreFrame.includeReaction[7]
            || SegreFrame.includeReaction[8] ) ) {

            mouseX = me.getX();
            mouseY = me.getY();
            getNZ(mouseX,mouseY);

            // Retoggle the boolean array indicating whether
            // the isotope is selected or not (getNZ toggles
            // this array, but for this case (ctrl-press) we
            // don't want it to be toggled, so toggle second
            // time to undo the first one).

            if(protonNumber >= 0) {
                isoColor[protonNumber][neutronNumber] =
                    ! isoColor[protonNumber][neutronNumber];
            }
            

            if( me.isControlDown() ) {
                id = new IsoData(600,600," Isotope Data","text2");

                // Don't show until mouserelease event below to get around bug
                // in Windows where the window ends up behind the isotope display
                // if it is shown here.
    
                id.show();

            } else if ( isoColor[protonNumber][neutronNumber] ){
                ad = new AbundanceData(505,360," Set Abundance Data", "");

                // Don't show until mouserelease event below to get around bug
                // in Windows where the window ends up behind the isotope display
                // if it is shown here.
    
                ad.show();
            }

        // If ctrl or shift not down on press and at least one
        // reaction class has been selected

        } else if (   SegreFrame.includeReaction[1]
            || SegreFrame.includeReaction[2]
            || SegreFrame.includeReaction[3]
            || SegreFrame.includeReaction[4]
            || SegreFrame.includeReaction[5]
            || SegreFrame.includeReaction[6]
            || SegreFrame.includeReaction[7]
            || SegreFrame.includeReaction[8] ) {

            Graphics g = this.getGraphics();
            mouseX = me.getX();
            mouseY = me.getY();
            getNZ(mouseX,mouseY);

            // Set the reaction array entries for this isotope
            // to true or false, depending on which reaction
            // categories are toggled and whether the isotope
            // is selected or not.

            for (int i=1; i<=8; i++) {
                if ( SegreFrame.includeReaction[i]
                        && isoColor[protonNumber][neutronNumber] ) {
                    DataHolder.includeReaction[protonNumber]
                        [neutronNumber][i] = true;
                } else {
                    DataHolder.includeReaction[protonNumber]
                        [neutronNumber][i] = false;
                }
            }

            // Set the appropriate color for the square

            Color theColor;
            if(isAbundant[protonNumber][neutronNumber]
                && isoColor[protonNumber][neutronNumber]) { // has initial abundance
                theColor = initAbundColor;
            } else if(isoColor[protonNumber][neutronNumber]) { // include in network
                theColor = selectColor;
            } else {                                           // exclude from network
                theColor = nonSelectColor;
                isAbundant[protonNumber][neutronNumber] = false;

                if (protonNumber==1 && neutronNumber==0) {
                    StochasticElements.Y[1][0]=StochasticElements.YH=0.0;
                } else if (protonNumber==2 && neutronNumber==1) {
                    StochasticElements.Y[2][1]=0.0;
                } else if (protonNumber==2 && neutronNumber==2) {
                    StochasticElements.Y[2][2]=StochasticElements.YHe=0.0;
                } else {

                    int removeIndex = 0;
                    for (int i=0; i<StochasticElements.seedY.length; i++) {
                        if(StochasticElements.seedProtonNumber[i]==protonNumber
                            && StochasticElements.seedNeutronNumber[i]==neutronNumber){
                            StochasticElements.seedY[i] = 0.0;
                            removeIndex = i;
                        }
                    }
                    StochasticElements.Y[protonNumber][neutronNumber] = 0.0;
                }
            }

            // Draw the box (unless not particle stable)

            if(isPStable[protonNumber][neutronNumber]) {
                drawColorSquare(xoffset+neutronNumber*boxWidth,
                    yoffset+(zmax-protonNumber)*boxHeight,
                    boxWidth,boxHeight,
                    theColor,frameColor,ig);
                if(showIsoLabels) {
                    setIsoLabel(protonNumber,neutronNumber,ig);
                }

            }

            // Following repaint necessary to display the new square
            // just written to the Graphics buffer ig.  Repaint only
            // the square changed to prevent repainting of entire screen
            // and resulting flash.


            repaint(xoffset+neutronNumber*boxWidth,
                yoffset+(zmax-protonNumber)*boxHeight,
                boxWidth,boxHeight);


            // Issue warning and do no selection if no reaction classes
            // have been chosen
    
        } else {
    
            // The method makeTheWarning requires as its last argument
            // the name of a Frame that is the modally blocked window
            // when the warning dialog is displayed (the constructor for a
            // Dialog requires as one argument a Frame---or a Dialog---that
            // is the modally blocked parent window).  If invoked from
            // a class extending Frame, the required argument can be just
            // "this".  Since the present class is a Canvas, we must specify
            // a different Frame.  In this case, the logical choice is
            // the static instance cd of SegreFrame created in the
            // class ChooseIsotopes, since that is the window containing
            // the IsotopePad canvas.
    
                makeTheWarning(300,300,210,100,Color.black,
                    MyColors.warnColorBG, " Warning!",
                    "Select Reaction Classes First!", true ,
                    ChooseIsotopes.cd);
        }

    }

    public void mouseEntered(MouseEvent me) {   // Not used
    }

    public void mouseExited(MouseEvent me) {    // Not used

    }

    public void mouseClicked(MouseEvent me) {

    // The following commented out code would implement a
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

    public void mouseReleased(MouseEvent me) { 
        if(me.isShiftDown() && ad != null){
           // ad.show();
        }
        if(me.isControlDown() && id != null){
            //id.show();
        }
    }



    // ------------------------------------------------------------------------------------------
    //  Method getNZ to determine the N and Z corresponding to
    //  the square clicked
    // ------------------------------------------------------------------------------------------

    public void getNZ(int x, int y) {

        // Return immediately if outside bounds of the chart
        if(x < xoffset || x > xmax || y < yoffset || y > ymax) {
            return;
        // Otherwise determine the N and Z of the clicked square if between the drip lines
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

                // Toggle the boolean array indicating whether
                // the isotope is selected or not
                if( protonNumber >= 0
                    && isPStable[protonNumber][neutronNumber] ) {
                    isoColor[protonNumber][neutronNumber] =
                        ! isoColor[protonNumber][neutronNumber];
                }
            }
        }
    }



    // ----------------------------------------------------------------------------------------------
    //  Method drawColorSquare to draw colored square with outline
    // ----------------------------------------------------------------------------------------------

    void drawColorSquare(int x, int y, int delx, int dely,
        Color bgcolor, Color frameColor, Graphics g) {
        //if(protonNumber == 0){return;}
        g.setColor(bgcolor);
        g.fillRect(x,y,delx,dely);
        g.setColor(frameColor);
        g.drawRect(x,y,delx,dely);
    }


// -------------------------------------------------------------------------------------------
//  Method setIsoLabel to set labels for individual isotopes
// -------------------------------------------------------------------------------------------

    void setIsoLabel(int z, int n, Graphics g) {
        if(protonNumber == 0){return;}
        String tempS = returnSymbol(z);
        String tempS2 = String.valueOf(z+n);
        int wid = realSmallFontMetrics.stringWidth(tempS)
            + tinyFontMetrics.stringWidth(tempS2);
        int xzero = xoffset+n*boxWidth+boxWidth/2-wid/2;
        int yzero = yoffset+(zmax-z+1)*boxHeight-boxHeight/2+1;
        g.setFont(tinyFont);
        g.setColor(isoLabelColor);
        g.drawString(tempS2,xzero,yzero);        // Symbol
        xzero += tinyFontMetrics.stringWidth(tempS2);
        yzero += 5;
        g.setFont(realSmallFont);
        g.drawString(tempS,xzero,yzero);         // Mass Number
    }



    // ----------------------------------------------
    //  paint method
    // ----------------------------------------------

    public void paint(Graphics g){

        if (image == null) {
            image = createImage(getSize().width+1000, getSize().height+1000);
            ig = image.getGraphics();
            drawMesh(xoffset,yoffset,boxWidth,boxHeight,ig);
        }

        // Following used to get square to update
        // to right color after initial abundance set

        if(updateAbundance){
            drawMesh(xoffset,yoffset,boxWidth,boxHeight,ig);
            updateAbundance = false;
        }
        g.drawImage(image,0,0,null);
    }


    // ------------------------------------------------------------------------------------------
    //  The method PSfile generates a postscript file of the
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
    //  a Windows file system.
    // -------------------------------------------------------------------------------------------

    public void PSfile (String fileName) {

        try {
            FileWriter psOut = new FileWriter(fileName);
            Graphics postscript = new PSGr1(psOut);
            drawMesh(this.xoffset,this.yoffset,this.boxWidth, this.boxHeight,postscript);
    
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
    
        } catch (Exception e) {System.out.println(e);}
    }



    // ----------------------------------------------------------------------------------------------
    //  Method processSquares to process square when it is selected
    //  for the network. Used in conjunction with the
    //  select all squares button.
    // ----------------------------------------------------------------------------------------------

    public void processSquares(int Z, int N) {

        // Set the reaction array entries for this isotope
        // to true or false, depending on which reaction
        // categories are toggled and whether the isotope
        // is selected or not.

        for (int i=1; i<=8; i++) {
            if ( SegreFrame.includeReaction[i]
                && isoColor[Z][N] ) {
                DataHolder.includeReaction[Z][N][i] = true;
            } else {
                DataHolder.includeReaction[Z][N][i] = false;
            }
        }

        // Set the appropriate color for the square

        Color theColor;
        if(isAbundant[Z][N]){               // initial abundance
            theColor = initAbundColor;
        } else if(isoColor[Z][N]) {        // include in network
            theColor = selectColor;
        } else {                                 // exclude from network
            theColor = nonSelectColor;
        }

        // Draw the box (unless not particle stable)

        if(isPStable[Z][N]) {
            drawColorSquare(xoffset+N*boxWidth,
                yoffset+(zmax-Z)*boxHeight,
                boxWidth,boxHeight,
                theColor,frameColor,ig);

            if(showIsoLabels) {
                setIsoLabel(Z,N,ig);
            }
        }

        // Following repaint necessary to display the new square
        // just written to the Graphics buffer ig.  Repaint only
        // the square changed to prevent repainting of entire screen
        // and resulting flash.

        repaint(xoffset+N*boxWidth,
            yoffset+(zmax-Z)*boxHeight,
            boxWidth,boxHeight);
    }


    // ----------------------------------------------------------------------------------------
    //  Method returnSymbol to return element symbol given the
    //  proton number.
    // ----------------------------------------------------------------------------------------

    static String returnSymbol (int z) {

        String [] symbolString = {"n","H","He","Li","Be","B","C","N",
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
    //  The method makeTheWarning creates a modal warning window when
    //  invoked from within an object that subclasses Frame. The window is
    //  modally blocked (the window from which the warning window is
    //  launched is blocked from further input until the warning window
    //  is dismissed by the user).  Method arguments:
    //
    //      X = x-position of window on screen (relative upper left)
    //      Y = y-position of window on screen (relative upper left)
    //      width = width of window
    //      height = height of window
    //      fg = foreground (font) color
    //      bg = background color
    //      title = title string
    //      text = warning string text
    //      oneLine = display as one-line Label (true)
    //                or multiline TextArea (false)
    //      frame = A Frame that is the parent window modally
    //              blocked by the warning window.  If the parent
    //              class from which this method is invoked extends
    //              Frame, this argument can be just "this" (or
    //              "ParentClass.this" if invoked from an
    //              inner class event handler of ParentClass).
    //              Otherwise, it must be the name of an object derived from
    //              Frame that represents the window modally blocked.
    //
    //  If oneLine is true, you must make width large enough to display all
    //  text on one line.
    // --------------------------------------------------------------------------------------------------------

    public void makeTheWarning (int X, int Y, int width, int height,
        Color fg, Color bg, String title,
        String text, boolean oneLine, Frame frame) {

        Font warnFont = new java.awt.Font("SanSerif", Font.BOLD, 12);
        FontMetrics warnFontMetrics = getFontMetrics(warnFont);

        // Create Dialog window with modal blocking set to true.
        // Make final so inner class below can access it.

        final Dialog mww = new Dialog(frame, title, true);
        mww.setLayout(new BorderLayout());
        mww.setSize(width,height);
        mww.setLocation(X,Y);

        // Use Label for 1-line warning

        if (oneLine) {
            Label hT = new Label(text,Label.CENTER);
            hT.setForeground(fg);
            hT.setBackground(bg);
            hT.setFont(warnFont);
            mww.add("Center", hT);

        // Use TextArea for multiline warning

        } else {
            TextArea hT = new TextArea("",height,width, TextArea.SCROLLBARS_NONE);
            hT.setEditable(false);
            hT.setForeground(fg);
            hT.setBackground(bg);     // no effect once setEditable (false)?
            hT.setFont(warnFont);
            mww.add("Center", hT);
            hT.appendText(text);
        }

        mww.setTitle(title);

        // Add dismiss button

        Panel botPanel = new Panel();
        botPanel.setBackground(Color.lightGray);
        Label label1 = new Label();
        Label label2 = new Label();

        Button dismissButton = new Button("Dismiss");
        botPanel.add(label1);
        botPanel.add(dismissButton);
        botPanel.add(label2);

        // Add inner class event handler for Dismiss button.  This must be
        // added to the dismissButton before botPanel is added to mww.

        dismissButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
            mww.hide();
            mww.dispose();
            }
        });

        mww.add("South", botPanel);

        // Add window closing button (inner class)

        mww.addWindowListener(new WindowAdapter() {
        public void windowClosing(WindowEvent e) {
            mww.hide();
            mww.dispose();
        }
        });

        mww.show();         // Note that this show must come after all the above
                                    // additions; otherwise they are not added before the
                                    // window is displayed.
    }

}  /* End class IsotopePad */

