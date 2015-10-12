// --------------------------------------------------------------------------------------------
//  Class SegreFrame to lay out main interface window
//  for isotope and reaction selection.  Required to implement
//  ItemListener because we are going to listen for changes
//  in Checkboxes and act accordingly.
// --------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;


class SegreFrame extends Frame implements ItemListener {

    static FileOutputStream toFileS;
    static PrintWriter toFile;

    Font titleFont = new java.awt.Font("SanSerif", Font.BOLD, 12);
    FontMetrics titleFontMetrics = getFontMetrics(titleFont);
    Font buttonFont = new java.awt.Font("SanSerif", Font.BOLD, 11);
    FontMetrics buttonFontMetrics = getFontMetrics(buttonFont);
    Font textFont = new java.awt.Font("SanSerif", Font.PLAIN, 12);
    FontMetrics textFontMetrics = getFontMetrics(textFont);

    static boolean [] includeReaction = new boolean [9];
    static boolean helpWindowOpen = false;
    MyHelpFrame hf = new MyHelpFrame();

    Color panelBackColor = MyColors.gray204;
    Color panelForeColor = MyColors.gray51;
    Color warnColorBG = MyColors.warnColorBG;

    // Index (0=small, 1=medium, 2=large) for current box size and constants
    // setting the dimension of corresponding boxes

    int currentSizeIndex = 0;
    static final int SMALLBOXSIZE = 12; //17;
    static final int MEDBOXSIZE = 26;  //29;
    static final int LARGEBOXSIZE = 33;

    static IsotopePad gp = new IsotopePad();
    static PlotParams pm = null;

    Panel cboxPanel = new Panel();

    final Checkbox checkBox1 = new Checkbox(" Class 1     ");
    final Checkbox checkBox2 = new Checkbox(" Class 2     ");
    final Checkbox checkBox3 = new Checkbox(" Class 3     ");
    final Checkbox checkBox4 = new Checkbox(" Class 4     ");
    final Checkbox checkBox5 = new Checkbox(" Class 5     ");
    final Checkbox checkBox6 = new Checkbox(" Class 6     ");
    final Checkbox checkBox7 = new Checkbox(" Class 7     ");
    final Checkbox checkBox8 = new Checkbox(" Class 8     ");
    final Checkbox checkBox9 = new Checkbox(" All Classes ");

    static ScrollPane sp;

    static ProgressMeter prom;
    static StochasticElements se;


    // ----------------------------------------------------------------
    //  Public constructor
    // ----------------------------------------------------------------

    public SegreFrame() {
        
        setLayout(new BorderLayout());
        
        // Create a graphics canvas of class IsotopePad and attach
        // it as a scrollable child to a ScrollPane
        
        sp = new ScrollPane(ScrollPane.SCROLLBARS_AS_NEEDED);
        sp.add(gp);
        this.add("Center", sp);
        
        // Define a GridBagConstraint object that we will reuse
        // several times in different GridBagLayouts
        
        GridBagConstraints gs = new GridBagConstraints();
        gs.weightx = 100;
        gs.weighty = 100;
        gs.gridx = 0;
        gs.gridy = 0;
        gs.ipadx = 0;
        gs.ipady = 0;
        gs.fill = GridBagConstraints.HORIZONTAL;
        gs.anchor = GridBagConstraints.NORTH;
        gs.insets = new Insets(10,6,0,0);
        
        Panel p = new Panel();
        p.setLayout(new GridBagLayout());
        p.setForeground(panelForeColor);
        p.setBackground(panelBackColor);
        
        Panel tPanel = new Panel();
        tPanel.setLayout(new GridLayout(1,1,5,5));
        tPanel.setFont(titleFont);
        Label reacLabel = new Label("Reaction Class",Label.LEFT);
        tPanel.add(reacLabel);
        tPanel.setForeground(panelForeColor);
        
        cboxPanel.setLayout(new GridLayout(10,1,8,8));
        
        // Add top label and checkboxes to the cboxPanel panel
        
        cboxPanel.add(tPanel);
        cboxPanel.add(checkBox1);
        cboxPanel.add(checkBox2);
        cboxPanel.add(checkBox3);
        cboxPanel.add(checkBox4);
        cboxPanel.add(checkBox5);
        cboxPanel.add(checkBox6);
        cboxPanel.add(checkBox7);
        cboxPanel.add(checkBox8);
        cboxPanel.add(checkBox9);
        
        // Add ItemListeners to all Checkboxes.  These will cause the
        // actions defined in the method ItemStateChanged to be
        // executed whenever a checkbox state is changed.
        
        checkBox1.addItemListener(this);
        checkBox2.addItemListener(this);
        checkBox3.addItemListener(this);
        checkBox4.addItemListener(this);
        checkBox5.addItemListener(this);
        checkBox6.addItemListener(this);
        checkBox7.addItemListener(this);
        checkBox8.addItemListener(this);
        checkBox9.addItemListener(this);

        // Set to include all reaction classes as default

        checkBox1.setState(true);
        checkBox2.setState(true);
        checkBox3.setState(true);        
        checkBox4.setState(true);
        checkBox5.setState(true);
        checkBox6.setState(true);
        checkBox7.setState(true);
        checkBox8.setState(true);

        for (int i=1; i<9; i++) {
            includeReaction[i] = true;
        }
        
        // Add the checkbox panel to Panel p
        
        p.add(cboxPanel, gs);
        
        // Panel to hold action buttons
        
        Panel goPanel = new Panel();
        goPanel.setLayout(new GridLayout(7,1,5,5));
        gs.insets = new Insets(7,10,0,10);
        gs.ipady = 2;
        
        // Create Clear button and add to goPanel
        
        Button clearButton = new Button("Clear");
        clearButton.setFont(buttonFont);
        gs.gridy = 0;
        goPanel.add(clearButton, gs);
        
        // Create Calculate button and add to goPanel
        
        Button goButton = new Button("Integrate");
        goButton.setFont(buttonFont);
        gs.gridy = 1;
        goPanel.add(goButton, gs);
        
        // Create button to launch parameter setting window and add to goPanel
        
        Button PSButton = new Button("Parameters");
        PSButton.setFont(buttonFont);
        gs.gridy = 2;
        goPanel.add(PSButton, gs);
        
        // Create rate plotting button and add to goPanel
        
        Button rateButton = new Button("Plot Rates");
        rateButton.setFont(buttonFont);
        gs.gridy = 3;
        goPanel.add(rateButton, gs);
        
        // Create plotOldButton and add to goPanel
        
        Button plotOldButton = new Button("Plot Old");
        plotOldButton.setFont(buttonFont);
        gs.gridy = 4;
        goPanel.add(plotOldButton, gs);
        
        // Create 3DPlotButton and add to goPanel
        
        Button threeDPlotButton = new Button("3D Plotter");
        threeDPlotButton.setFont(buttonFont);
        gs.gridy = 5;
        goPanel.add(threeDPlotButton, gs);
        
        // Create Help button and add to goPanel
        
        Button helpButton = new Button("Help");
        helpButton.setFont(buttonFont);
        gs.gridy = 5;
        goPanel.add(helpButton, gs);
        
        // Add the goPanel to the Panel p
        
        gs.anchor = GridBagConstraints.NORTH;
        gs.gridy = 1;
        p.add(goPanel, gs);
        goPanel.setForeground(panelForeColor);
        
        // Add the Panel p to the main layout
        
        this.add("East",p);
        
        // Bottom panel with widgets
        
        Panel botPanel = new Panel();
        botPanel.setLayout(new GridBagLayout());
        botPanel.setBackground(panelBackColor);
        botPanel.setForeground(panelForeColor);
        botPanel.setFont(buttonFont);
        
        // Reuse GridBagConstraints gs
        
        gs.weightx = 100;
        gs.weighty = 100;
        gs.gridx = 0;
        gs.gridy = 0;
        gs.ipadx = 0;
        gs.ipady = 0;
        gs.fill = GridBagConstraints.NONE;
        gs.anchor = GridBagConstraints.WEST;
        gs.insets = new Insets(4,10,4,0);
        
        Panel botPanelA = new Panel();
        Label boxSizeLabel = new Label("Box Size",Label.LEFT);
        botPanelA.add(boxSizeLabel);
        
        final Choice boxSize = new Choice();
        boxSize.setFont(textFont);
        boxSize.addItem("Small");
        boxSize.addItem("Medium");
        boxSize.addItem("Large");
        boxSize.select(0);            // Set small by default
        botPanelA.add(boxSize);
        gs.gridx = 0;
        botPanel.add(botPanelA, gs);
        
        Panel botPanelB = new Panel();
        Label zmaxLabel = new Label("Max Z",Label.LEFT);
        botPanelB.add(zmaxLabel);
        
        final TextField zmaxField = new TextField(3);
        zmaxField.setFont(textFont);
        zmaxField.setText(Integer.toString(IsotopePad.zmax));
        botPanelB.add(zmaxField);
        gs.insets = new Insets(4,0,4,0);
        gs.gridx = 1;
        botPanel.add(botPanelB, gs);
        
        Panel botPanelC = new Panel();
        Label nmaxLabel = new Label("Max N",Label.LEFT);
        botPanelC.add(nmaxLabel);
        
        final TextField nmaxField = new TextField(3);
        nmaxField.setFont(textFont);
        nmaxField.setText(Integer.toString(IsotopePad.nmax));
        botPanelC.add(nmaxField);
        gs.gridx = 2;
        botPanel.add(botPanelC, gs);
        
        Button resetButton = new Button("Reset");
        resetButton.setFont(buttonFont);
        gs.gridx = 3;
        botPanel.add(resetButton, gs);
        
        final Button isoButton = new Button("Show Labels");
        isoButton.setFont(buttonFont);
        gs.gridx = 4;
        botPanel.add(isoButton, gs);
        
        final Button allButton = new Button("Select Isotopes");
        allButton.setFont(buttonFont);
        gs.insets = new Insets(4,0,4,120);
        gs.gridx = 5;
        botPanel.add(allButton, gs);
        
        this.add("South",botPanel);
        
        // Add blank left and top panels for spacing
        
        Panel leftPanel = new Panel();
        leftPanel.setBackground(panelBackColor);
        leftPanel.setSize(10,this.getSize().height);
        
        this.add("West",leftPanel);
        
        Panel topPanel = new Panel();
        topPanel.setBackground(panelBackColor);
        topPanel.setSize(this.getSize().width,10);
        
        this.add("North",topPanel);
        
 
        // ------ Now define the button action handlers ------- //
        
        // Button action to process reactions.  Handle with an inner class
        
        goButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
        
                StochasticElements.doIntegrate = true;  
    
                // Create instance of StochasticElements to do
                // the calculation but do some elementary checks first
                // to be sure everything has been set that is required.  
        
                // Check to be sure start integration time is less than end integration time
                boolean startGTstop = false;
                if (StochasticElements.logtmax <= StochasticElements.logtmin){
                    startGTstop = true;
                }
        
                // Check to be sure start plot time is less than end plot time
                boolean startGTstop2 = false;
                if (StochasticElements.logtmaxPlot <= StochasticElements.logtminPlot){
                    startGTstop2 = true;
                }
            
                // Check to see that at least one initial abundance is non-zero
                boolean checkAbundances = false;
                for(int Z=0; Z<DataHolder.Znum; Z++){
                    for(int N=0; N<DataHolder.Nnum; N++){
                        if(IsotopePad.isAbundant[Z][N]){
                            checkAbundances = true;
                            break;
                        }
                    }
                }
                    
                if( StochasticElements.nintervals > (StochasticElements.tintMax-2) ) {
                    String screwup = "Max number of time intervals is ";
                    screwup += (StochasticElements.tintMax-2);
                    screwup += ". Click Set Parameters button and set an";
                    screwup += " appropriate value for the number of";
                    screwup += " time intervals.";
                    makeTheWarning(300,300,300,150,Color.black,warnColorBG, " Warning!",
                        screwup, false, SegreFrame.this);
                } else if (!StochasticElements.parametersWereSet){
                    String screwup = "You have not set any parameters!";
                    screwup += " Click the Set Parameters button to open a dialog to set";
                    screwup += " parameters, click Save for that dialog when complete, and then ";
                    screwup += " click Calculate again.";
                    makeTheWarning(300,300,200,180,Color.black,
                    warnColorBG, " Warning!",
                    screwup, false, SegreFrame.this);
                } else if (!checkAbundances){
                    String screwup = "You must set at least one non-zero abundance!";
                    screwup += " Shift-click on an active (purple) isotope to get";
                    screwup += " abundance-setting dialog, set initial abundances, and";
                    screwup += " click Calculate again.";
                    makeTheWarning(300,300,300,180,Color.black,
                    warnColorBG, " Warning!",
                    screwup, false, SegreFrame.this);
                } else if (startGTstop){
                    String screwup = "Starting time greater than";
                    screwup += " ending time for integration.";
                    screwup += " Click Set Parameters button and change.";
                    makeTheWarning(300,300,200,150,Color.black,warnColorBG, " Error!",
                        screwup, false, SegreFrame.this);
                } else if (startGTstop2){
                    String screwup = "Starting plot time greater than";
                    screwup += " ending plot time.";
                    screwup += " Click Set Parameters button and change.";
                    makeTheWarning(300,300,200,150,Color.black,warnColorBG, " Error!",
                        screwup, false, SegreFrame.this);
                } else if ( StochasticElements.totalSeeds > 0.0
                    && StochasticElements.Ye > 0.0
                    && StochasticElements.nintervals > 0
                    && ( StochasticElements.rho > 0.0
                    || StochasticElements.T9 > 0.0
                    || StochasticElements.profileFileName != "")
                    && StochasticElements.stochasticFactor > 0.0
                    && StochasticElements.pmax > 0
                    && StochasticElements.nmax > 0
                    && StochasticElements.pmin > 0 ) {
                    prom = new ProgressMeter(100,100,400,120,"", "Loading rates ...", "");
                    se = new StochasticElements();
                } else {
                    String screwup = "There are parameters not set.";
                    screwup += " Click the Set Parameters button and";
                    screwup += " specify required parameters first.";
                    makeTheWarning(300,300,200,150,Color.black,warnColorBG, " Warning!",
                        screwup, false, SegreFrame.this);    
                }
            }
        });
        

        // Button action to process plotting old data.  Handle with
        // an inner class
    
        plotOldButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){

            // Create instance of StochasticElements to read in and display old data set. It first
            // opens a dialog box that permits the user to specify the name of the input data file. 

            StochasticElements.write3DOutput = true;   
            StochasticElements.doIntegrate = false;

              OldFileDialogue fd = new OldFileDialogue(200,300,400,110,Color.black,
                Color.lightGray,"Choose File Name",
                "Choose name for input file:");
              fd.setResizable(false);
              fd.hT.setText(StochasticElements.oldYfile);
              fd.show();
        }
    });


     // Open 3d plotter

    threeDPlotButton.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ae){

            // Start the system command to open the 3D plotter using JNI
       
            Process proc;
            try {
                Runtime runtime = Runtime.getRuntime();
                String command = "java elementProto";
                proc = runtime.exec(command);         
            

                // Put a BufferedReader on the system error output to report back any errors
                // generated in trying to launch the 3D program.
    
                InputStream inputstream = proc.getErrorStream();         //proc.getInputStream();
                InputStreamReader inputstreamreader = new InputStreamReader(inputstream);
                BufferedReader bufferedreader = new BufferedReader(inputstreamreader);
        
                // Read and write any system command error output
        
                String allOfIt = ""; 
                String line;
                int lineCounter = 0;
                while ((line = bufferedreader.readLine()) != null) {
                    lineCounter ++;
                    allOfIt += line+"\n";
                }
                if(lineCounter > 0){
                    StochasticElements.callExit(
                        "\nProblem with the 3D java program elementProto.java:\n\n"+allOfIt);
                }
            }
            catch (IOException e) {System.out.println(e);}          
        }
    });


    // Help button actions.  Handle with an inner class.
    
    helpButton.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ae){
    
            if(helpWindowOpen) {
                hf.toFront();
            } else {
                hf = new MyHelpFrame();
                hf.setSize(300,400);
                hf.setLocation(100,100);
                hf.setResizable(false);
                hf.setTitle(" Help");
                hf.show();
                helpWindowOpen = true;
            }
    
            // Following example illustrates how to execute a standard
            // program from a Java program.  It is just put here because
            // this is a convenient button for the demo.
    
            //try {
            //    Process NP = Runtime.getRuntime().exec("C:\\WINNT\\NOTEPAD.EXE");
    
            //} catch (java.io.IOException e) {;}
    
        }
    });

    
    // Rates button actions.  Handle with an inner class.
    
    rateButton.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ae){
    
            // Open window to specify plot parameters
    
            pm = new PlotParams(270,450, " Plotting Parameters", "");
            pm.setLocation(570,50);
            pm.show();
        }
    });
    
    
    //  Launch window to set up parameters for the calculation.
    //  Handle with inner class.
    
    PSButton.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ae){
            ParamSetup pset = new ParamSetup(490,700," Parameter Setup","text2");
            pset.show();
            StochasticElements.parametersWereSet = true;
        }
    });
    
    
    // Clear button actions.  Handle with an inner class.
    
    clearButton.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ae){
    
            // Set the reaction array to false
    
            for (int i=1; i<=8; i++){includeReaction[i] = false;}
    
            // Uncheck the checkboxes
    
            checkBox1.setState(false);
            checkBox2.setState(false);
            checkBox3.setState(false);
            checkBox4.setState(false);
            checkBox5.setState(false);
            checkBox6.setState(false);
            checkBox7.setState(false);
            checkBox8.setState(false);
            checkBox9.setState(false);
        }
    });
    

    // Actions for Reset button.  Handle with an inner class

    resetButton.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ae){

            if( (byte)stringToInt(zmaxField.getText()) > StochasticElements.pmax 
                || (byte)stringToInt(nmaxField.getText()) > StochasticElements.nmax ){

                String message = "Largest Max Z can't be greater than ";
                message += StochasticElements.pmax;
                message += " (set by pmax in StochasticElements)";
                message+= " and largest Max N can't be greater than ";
                message += StochasticElements.nmax;
                message += " (set by nmax in the class StochasticElements).";
                message += " Change Max Z and/or Max N entries to conform, or";
                message += " change pmax or nmax in StochasticElements.";
                message += " There must also exist files in the date subdirectory";
                message += " isoZ_N.ser corresponding to the ranges of Z and N chosen.";
                makeTheWarning(300,300,250,250,Color.black,
                    warnColorBG, " Error!", message,
                    false, SegreFrame.this);
                return;
            }

            int index = boxSize.getSelectedIndex();
            if(index == 0 && gp.showIsoLabels){
                String message = "Can't use small boxes if isotope";
                message += " labels are displayed.  Either turn off";
                message += " isotope labels, or set size to medium";
                message += " or large.";
    
                makeTheWarning(300,300,200,150,Color.black,
                    warnColorBG, " Warning!",
                    message, false, SegreFrame.this );
                return;
            } else if (index == 0){
                gp.boxWidth = gp.boxHeight = SMALLBOXSIZE;   // small boxes
                currentSizeIndex = 0;
            } else if (index == 1){
                gp.boxWidth = gp.boxHeight = MEDBOXSIZE;      // medium boxes
                currentSizeIndex = 1;
            } else if (index == 2){
                gp.boxWidth = gp.boxHeight = LARGEBOXSIZE;   // large boxes
                currentSizeIndex = 2;
            }

            // Clear the existing plot from the Graphics buffer by
            // overwriting with the background color
    
            gp.ig.setColor(ChooseIsotopes.segreBC);
            gp.ig.fillRect(0,0,gp.xmax + gp.xoffset,gp.ymax + gp.yoffset);
    
            String zstring = zmaxField.getText();
            String nstring = nmaxField.getText();
            gp.zmax = stringToInt(zstring);
            gp.nmax = stringToInt(nstring);
            gp.width = gp.boxWidth*(gp.nmax+1);
            gp.height = gp.boxHeight*(gp.zmax+1);
            gp.setSize(gp.width+2*gp.xoffset,gp.height+2*gp.yoffset);
            gp.xmax = gp.xoffset + gp.width;
            gp.ymax = gp.yoffset + gp.height;
            gp.initPStable();

            // Update plot ranges for Z and N in StochasticElements

            StochasticElements.pmaxPlot = (byte)stringToInt(zstring);
            StochasticElements.nmaxPlot = (byte)stringToInt(nstring);
            StochasticElements.pmax = (byte)stringToInt(zstring);
            StochasticElements.nmax = (byte)stringToInt(nstring);

            // Write the current plot to the Graphics buffer
    
            gp.drawMesh(gp.xoffset,gp.yoffset,gp.boxWidth,gp.boxHeight,gp.ig);
    
            // Force redisplay to toggle scrollbar state
            // on scrollPane viewport if it has changed with
            // reset of zmax and nmax. Requires call to a method of outer
            // class because we are presently in an inner
            // class and this.show() is not recognized here.
            // Instead, to get a reference to "this" for the
            // outer class from an inner class, use the construction
            // "outerClassName.this" instead of "this"
    
            SegreFrame.this.show();
    
            // Reset scrollbars (if displayed) so that Z=0, N=0 is in
            // the lower left corner. Not sure how to determine the
            // coordinate for vertical scrollbar to position it at the
            // bottom, so make it larger (5000) than the vertical dimension
            // of the segre chart would ever be.
    
            sp.setScrollPosition(0,5000);

        }
    });  // -- end inner class for Reset button processing



    // Actions for isoButton button.  Handle with an inner class

    isoButton.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ae){
            if(gp.showIsoLabels){
                isoButton.setLabel("Show Labels");
                gp.showIsoLabels = false;
            } else {
                if(currentSizeIndex == 0){
                    String message = "Can't use small boxes if isotope";
                    message += " labels are displayed.  Either turn off";
                    message += " isotope labels, or set size to medium";
                    message += " or large.";

                    makeTheWarning(300,300,200,150,Color.black,
                            warnColorBG, " Warning!",
                            message, false, SegreFrame.this);
                    return;
                }
                isoButton.setLabel("Hide Labels");
                gp.showIsoLabels = true;
            }

            // Write the current plot to the Graphics buffer
            gp.drawMesh(gp.xoffset,gp.yoffset,gp.boxWidth,gp.boxHeight,gp.ig);
        }
    });  // -- end inner class for isoButton button processing



    // Actions for allButton button (Isotope selection).  Handle with an inner class

    allButton.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ae){

            // Allow selection of isotopes for the reaction network, but
            // check first that at least one reaction class has
            // been chosen.

            if ( SegreFrame.includeReaction[1]
                || SegreFrame.includeReaction[2]
                || SegreFrame.includeReaction[3]
                || SegreFrame.includeReaction[4]
                || SegreFrame.includeReaction[5]
                || SegreFrame.includeReaction[6]
                || SegreFrame.includeReaction[7]
                || SegreFrame.includeReaction[8] ) {

                // Open a window that allows choice of how isotopes are to be selected

                ChooseActiveIsotopes cai = new ChooseActiveIsotopes(430,250,
                    " Choose Active Isotopes","text2");
                cai.show();

            // Issue warning and do no selection if no reaction classes
            // have been chosen

            } else {

                makeTheWarning(300,300,250,100,Color.black,
                    warnColorBG, " Warning!",
                    "Select Reaction Classes First!",
                    true, SegreFrame.this);
            }
        }
    });  // -- end inner class for allButton button processing


    // Add window closing button (inner class)

    this.addWindowListener(new WindowAdapter() {
        public void windowClosing(WindowEvent e) {
        System.exit(0);   // window closing button
        }
    });

    }  /* End SegreFrame constructor method */



    // ---------------------------------------------------------------------------------------------------------
    //  Static method makePS invoked from MyFileDialogue instance to write
    //  postscript file of Segre plane.  gp is not recognized
    //  directly from that object, but by declaring the instances
    //  of SegreFrame and IsotopePad to be static, methods of gp
    //  can be invoked indirectly through this method.  Likewise
    //  for the method makeRepaint invoked from AbundanceData.
    // ---------------------------------------------------------------------------------------------------------

    static void makePS(String file) {
        gp.PSfile(file);
    }


    // Launches instance of StochasticElements to plot old file read in rather than
    // performing new integration. This is invoked from the class OldFileDialogue.

    static void launchOldPlot(){
    
        StochasticElements.write3DOutput = true;   
        StochasticElements.doIntegrate = false;
        se = new StochasticElements();
    }


    // ---------------------------------------------------------------------------------------------------
    //  Static method makeRepaint invoked from AbundanceData to force
    //  reset of box color when an initial abundance is saved.
    // ---------------------------------------------------------------------------------------------------

    static void makeRepaint() {
       IsotopePad.updateAbundance = true;
    }



    // ---------------------------------------------------------------------------------------------------------
    //  Method itemStateChanged to act when state of Checkboxes changes.
    //  Requires that the class (SegreFrame) implement the
    //  ItemListener interface, which in turn requires that the
    //  method itemStateChanged be defined explicitly since
    //  ItemListener is abstract.
    // ---------------------------------------------------------------------------------------------------------

    public void itemStateChanged(ItemEvent check) {

        // Process the reaction class checkboxes.  First
        // get the components of the panel cboxPanel (which
        // contains a label and 9 checkboxes) and store in
        // a Component array (Note: the method getComponents()
        // is inherited from the Container class by the
        // subclass Panel).

        Component [] components = cboxPanel.getComponents();

        // Now process these components.  First cast each
        // Component to a Checkbox.  Then use the getState()
        // method of Checkbox to return boolean true if
        // checked and false otherwise.  Use this to set the
        // values of the boolean array includeReaction[].
        // (Note: we must skip i=0 in the loop because the first
        // component in cboxPanel is a Label, not a Checkbox.)

        boolean checkAll = ((Checkbox)components[components.length-1]).getState();
        if ( checkAll ) {             // if "Choose All" selected
            for (int i=1; i<9; i++) {
                includeReaction[i] = true;
            }
            checkBox1.setState(true);
            checkBox2.setState(true);
            checkBox3.setState(true);
            checkBox4.setState(true);
            checkBox5.setState(true);
            checkBox6.setState(true);
            checkBox7.setState(true);
            checkBox8.setState(true);
        } else {
            for (int i=1; i<components.length-1; i++) {
                Checkbox cb = (Checkbox)components[i];
                includeReaction[i] = cb.getState();  // true or false
            }
        }
    }


    // -----------------------------------------------------------------------------------------------
    //  Static method stringToDouble to convert a string to a double
    // -----------------------------------------------------------------------------------------------

    static double stringToDouble (String s) {
        Double mydouble=Double.valueOf(s);    // String to Double (object)
        return mydouble.doubleValue();              // Return primitive double
    }


    // -----------------------------------------------------------------------------------
    //  Static method stringToInt to convert a string to an int
    // -----------------------------------------------------------------------------------

    static int stringToInt (String s) {
        Integer myInt=Integer.valueOf(s);     // String to Integer (object)
        return myInt.intValue();                    // Return primitive int
    }



    // ----------------------------------------------------------------------------------------------
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
    //  (You can use the PSfile method to output to a .ps file.)
    //
    //      Variables:
    //      xoff = horizontal offset from upper left corner
    //      yoff = vertical offset from upper left corner
    //      makeBorder = turn rectangular border on or off
    // ------------------------------------------------------------------------------------------------

    public void printThisFrame(int xoff, int yoff, boolean makeBorder) {

        java.util.Properties printprefs = new java.util.Properties();
        Toolkit toolkit = this.getToolkit();
        PrintJob job = toolkit.getPrintJob(this,"Java Print",printprefs);
        if (job == null) {return;}
        Graphics g = job.getGraphics();
        g.translate(xoff,yoff);          // Offset from upper left corner
        Dimension size = this.getSize();
        if (makeBorder) {                // Rectangular border
            g.drawRect(-1,-1,size.width+2,size.height+2);
        }
        g.setClip(0,0,size.width,size.height);
        this.printAll(g);

        g.dispose();
        job.end();
    }



    // --------------------------------------------------------------------------------------------------------------------
    //  The method makeTheWarning creates a modal warning window when invoked
    //  from within an object that subclasses Frame. The window is
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
    // -----------------------------------------------------------------------------------------------------------------------

    public void makeTheWarning (int X, int Y, int width, int height,
        Color fg, Color bg, String title, String text, boolean oneLine, Frame frame) {

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
            hT.setBackground(bg);  // no effect once setEditable (false)?
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

        mww.show();     // Note that this show must come after all the above
                                // additions; otherwise they are not added before the
                                // window is displayed.
    }

}  /* End class SegreFrame */

