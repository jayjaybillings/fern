// ---------------------------------------------------------------------------------------------------------
//  Class AbundanceData to pop up window to input abundances for initial
//  seed isotopes (triggered by shift-mousepress on isotope square).
//  Offers 5 options for reading in data:  (1) input mass fraction X(i)
//  for this isotope, (2) input molar abundance Y(i) for isotope,
//  (3) read all abundances (this isotope and others) in from a file,
//  (4) choose the tabulated Solar abundance for this isotope only,
//  and (5) choose the tabulated Solar abundances for initial
//  abundances of all isotopes in the calculation.
// ---------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;


class AbundanceData extends Frame implements ItemListener {

    static boolean thisHelpWindowOpen = false;
    GenericHelpFrame hf = new GenericHelpFrame("","",0,0,0,0,0,0);

    static final double LOG10 = 0.434294482;     //  Conversion ln to log10
    static GraphicsGoodies2 gg = new GraphicsGoodies2();

    Font titleFont = new java.awt.Font("SanSerif", Font.BOLD, 12);
    FontMetrics titleFontMetrics = getFontMetrics(titleFont);
    Font buttonFont = new java.awt.Font("SanSerif", Font.BOLD, 11);
    FontMetrics buttonFontMetrics = getFontMetrics(buttonFont);
    Font textFont = new java.awt.Font("SanSerif", Font.PLAIN, 12);
    FontMetrics textFontMetrics = getFontMetrics(textFont);

    Color panelForeColor = Color.black;
    Color panelBackColor = new Color(230,230,230);
    Color disablebgColor = new Color(220,220,220);
    Color disablefgColor = new Color(180,180,180);
    Color textfieldColor = new Color(255,255,255);

    CheckboxGroup cbg = new CheckboxGroup();
    final Checkbox [] checkBox = new Checkbox[5];
    Panel panel0, panel1, panel2, panel3, panel4;
    Panel cboxPanel;
    TextField massFrac,Y,solarAbund;
    static TextField fileName;
    Label massFracL, YL, solarAbundL, fileNameL;

    int ZZ, NN;
    static String s;
    static int buffNumber;


    // ---------------------------------------------------------------------
    //  Public constructor
    // ---------------------------------------------------------------------

    public AbundanceData (int width, int height, String title, String text){

        String nn = "   (Z = "+String.valueOf(IsotopePad.protonNumber);
        nn += "   N = "+String.valueOf(IsotopePad.neutronNumber)+")";
        String mass = String.valueOf(IsotopePad.protonNumber
                    + IsotopePad.neutronNumber);

        ZZ = IsotopePad.protonNumber;
        NN = IsotopePad.neutronNumber;

        this.pack();
        this.setSize(width,height);
        this.setTitle(title);

        // Create five checkboxes

        String temp = "Use Solar Abundance for ALL currently selected isotopes     ";
        checkBox[0] = new Checkbox("Specify Mass Fraction");
        checkBox[1] = new Checkbox("Specify Molar Abundance");
        checkBox[2] = new Checkbox("Read All Abundances");
        checkBox[3] = new Checkbox("Use Solar Abundance");
        checkBox[4] = new Checkbox(temp);

        // Make them part of a checkbox group (exclusive radio buttons)

        checkBox[0].setCheckboxGroup(cbg);
        checkBox[1].setCheckboxGroup(cbg);
        checkBox[2].setCheckboxGroup(cbg);
        checkBox[3].setCheckboxGroup(cbg);
        checkBox[4].setCheckboxGroup(cbg);

        // Add itemListeners to listen for checkbox events.  These events
        // will be processed by the method itemStateChanged

        checkBox[0].addItemListener(this);
        checkBox[1].addItemListener(this);
        checkBox[2].addItemListener(this);
        checkBox[3].addItemListener(this);
        checkBox[4].addItemListener(this);

        // Define a GridBagConstraint object for GridBagLayout
        // manager used below.

        GridBagConstraints gs = new GridBagConstraints();
        gs.weightx = 100;
        gs.weighty = 100;
        gs.gridx = 0;
        gs.gridy = 0;
        gs.ipadx = 0;
        gs.ipady = 0;
        gs.gridwidth = 1;
        gs.gridheight = 1;
        gs.fill = GridBagConstraints.NONE;
        gs.anchor = GridBagConstraints.NORTH;
        gs.insets = new Insets(8,8,8,8);
        gs.insets = new Insets(10,10,10,10);

        // Main container

        cboxPanel = new Panel();
        cboxPanel.setLayout(new GridBagLayout());
        cboxPanel.setBackground(panelBackColor);

        // Top label

        Label reacLabel = new Label("Seed Abundances "+nn,Label.CENTER);
        reacLabel.setFont(titleFont);
        reacLabel.setForeground(panelForeColor);
        gs.gridwidth = 2;
        cboxPanel.add(reacLabel,gs);

        // Mass fraction widgets

        gs.anchor = GridBagConstraints.WEST;
        gs.gridwidth = 1;
        gs.gridy = 1;
        cboxPanel.add(checkBox[0],gs);

        panel0 = new Panel();
        panel0.setFont(textFont);
        panel0.setForeground(panelForeColor);

        massFracL = new Label("Mass Fraction X(i)",Label.LEFT);
        panel0.add(massFracL);

        massFrac = new TextField(13);
        massFrac.setFont(textFont);
        temp = StochasticElements.profileFileName;
        massFrac.setText(temp);
        panel0.add(massFrac);

        gs.gridx = 1;
        gs.anchor = GridBagConstraints.EAST;
        cboxPanel.add(panel0,gs);

        // Abundance Y(i) widgets

        gs.gridx = 0;
        gs.gridy = 2;
        gs.anchor = GridBagConstraints.WEST;
        cboxPanel.add(checkBox[1], gs);

        panel1 = new Panel();
        panel1.setFont(textFont);
        panel1.setForeground(panelForeColor);

        YL = new Label("Abundance Y(i)",Label.LEFT);
        panel1.add(YL);

        Y = new TextField(13);
        Y.setFont(textFont);
        temp = StochasticElements.profileFileName;
        Y.setText(temp);
        panel1.add(Y);

        gs.gridx = 1;
        gs.anchor = GridBagConstraints.EAST;
        cboxPanel.add(panel1, gs);

        // Read from file widgets

        gs.gridx = 0;
        gs.gridy = 3;
        gs.anchor = GridBagConstraints.WEST;
        cboxPanel.add(checkBox[2], gs);

        panel2 = new Panel();
        panel2.setFont(textFont);
        panel2.setForeground(panelForeColor);

        fileNameL = new Label("File",Label.LEFT);
        panel2.add(fileNameL);

        fileName = new TextField(22);
        fileName.setFont(textFont);
        temp = StochasticElements.profileFileName;
        fileName.setText(temp);
        panel2.add(fileName);

        gs.gridx = 1;
        gs.anchor = GridBagConstraints.EAST;
        cboxPanel.add(panel2, gs);

        // Set solar abundance for just this isotope widgets

        gs.gridx = 0;
        gs.gridy = 4;
        gs.anchor = GridBagConstraints.WEST;
        cboxPanel.add(checkBox[3], gs);

        panel3 = new Panel();
        panel3.setFont(textFont);
        panel3.setForeground(panelForeColor);

        solarAbundL = new Label("Solar Abundance Y(i)",Label.LEFT);
        panel3.add(solarAbundL);

        solarAbund = new TextField(13);
        solarAbund.setFont(textFont);
        temp = StochasticElements.profileFileName;
        solarAbund.setText(temp);
        panel3.add(solarAbund);

        gs.gridx = 1;
        gs.anchor = GridBagConstraints.EAST;
        cboxPanel.add(panel3, gs);

        // Set solar abundance for ALL isotopes checkbox

        gs.gridx = 0;
        gs.gridy = 5;
        gs.gridwidth = 2;
        gs.anchor = GridBagConstraints.WEST;
        cboxPanel.add(checkBox[4], gs);

        // Add the main container to the frame

        this.add(cboxPanel,"Center");

        // Read any values already set and place in fields

        String temp2;
        if(StochasticElements.Y[ZZ][NN] != 0) {
            temp = StochasticElements.gg.decimalPlace(8,(StochasticElements.Y[ZZ][NN]));
            temp2 = StochasticElements.gg.decimalPlace(8,(StochasticElements.Y[ZZ][NN]
                *(double)(ZZ + NN)));
        } else {temp2 = temp = "";}
        Y.setText(temp);
        massFrac.setText(temp2);
        fileName.setText(StochasticElements.abundFileName);
        solarAbund.setText(Double.toString(SolarAbundances.sab[ZZ][NN]));

        // Set up the initial conditions of the window

        if( StochasticElements.abundPref == 0 ) {    // Input as mass fraction X
            checkBox[0].setState(true);
            checkBox[0].setForeground(panelForeColor);
            checkBox[1].setForeground(disablefgColor);
            checkBox[2].setForeground(disablefgColor);
            checkBox[3].setForeground(disablefgColor);
            checkBox[4].setForeground(disablefgColor);
            massFrac.enable();
            massFrac.requestFocus();
            massFrac.setBackground(textfieldColor);
            massFracL.setForeground(panelForeColor);
            Y.disable();
            Y.setBackground(disablebgColor);
            YL.setForeground(disablefgColor);
            fileName.disable();
            fileName.setBackground(disablebgColor);
            fileNameL.setForeground(disablefgColor);
            solarAbund.disable();
            solarAbund.setBackground(disablebgColor);
            solarAbundL.setForeground(disablefgColor);

        } else if ( StochasticElements.abundPref == 1 ) {  // Input as abundance Y
            checkBox[1].setState(true);
            checkBox[0].setForeground(disablefgColor);
            checkBox[1].setForeground(panelForeColor);
            checkBox[2].setForeground(disablefgColor);
            checkBox[3].setForeground(disablefgColor);
            checkBox[4].setForeground(disablefgColor);
            massFrac.disable();
            massFrac.setBackground(disablebgColor);
            massFracL.setForeground(disablefgColor);
            Y.enable();
            Y.setBackground(textfieldColor);
            Y.requestFocus();
            YL.setForeground(panelForeColor);
            fileName.disable();
            fileName.setBackground(disablebgColor);
            fileNameL.setForeground(disablefgColor);
            solarAbund.disable();
            solarAbund.setBackground(disablebgColor);
            solarAbundL.setForeground(disablefgColor);

        } else if ( StochasticElements.abundPref == 2 ) {   // From input file
            checkBox[2].setState(true);
            checkBox[0].setForeground(disablefgColor);
            checkBox[1].setForeground(disablefgColor);
            checkBox[2].setForeground(panelForeColor);
            checkBox[3].setForeground(disablefgColor);
            checkBox[4].setForeground(disablefgColor);
            massFrac.disable();
            massFrac.setBackground(disablebgColor);
            massFracL.setForeground(disablefgColor);
            Y.disable();
            Y.setBackground(disablebgColor);
            YL.setForeground(disablefgColor);
            fileName.enable();
            fileName.setBackground(textfieldColor);
            fileName.requestFocus();
            fileNameL.setForeground(panelForeColor);
            solarAbund.disable();
            solarAbund.setBackground(disablebgColor);
            solarAbundL.setForeground(disablefgColor);

        } else if ( StochasticElements.abundPref == 3 ) {  // Solar for this one 
            checkBox[3].setState(true);
            checkBox[3].requestFocus();
            checkBox[0].setForeground(disablefgColor);
            checkBox[1].setForeground(disablefgColor);
            checkBox[2].setForeground(disablefgColor);
            checkBox[3].setForeground(panelForeColor);
            checkBox[4].setForeground(disablefgColor);
            massFrac.disable();
            massFrac.setBackground(disablebgColor);
            massFracL.setForeground(disablefgColor);
            Y.disable();
            Y.setBackground(disablebgColor);
            YL.setForeground(disablefgColor);
            fileName.disable();
            fileName.setBackground(disablebgColor);
            fileNameL.setForeground(disablefgColor);
            solarAbund.disable();
            solarAbund.setBackground(textfieldColor);
            solarAbundL.setForeground(panelForeColor);

        } else if ( StochasticElements.abundPref == 4 ) {  // Solar for all
            checkBox[4].setState(true);
            checkBox[4].requestFocus();
            checkBox[0].setForeground(disablefgColor);
            checkBox[1].setForeground(disablefgColor);
            checkBox[2].setForeground(disablefgColor);
            checkBox[3].setForeground(disablefgColor);
            checkBox[4].setForeground(panelForeColor);
            massFrac.disable();
            massFrac.setBackground(disablebgColor);
            massFracL.setForeground(disablefgColor);
            Y.disable();
            Y.setBackground(disablebgColor);
            YL.setForeground(disablefgColor);
            fileName.disable();
            fileName.setBackground(disablebgColor);
            fileNameL.setForeground(disablefgColor);
            solarAbund.disable();
            solarAbund.setBackground(disablebgColor);
            solarAbundL.setForeground(disablefgColor);
        }

        // Put stuff on the bottom panel

        Panel botPanel = new Panel();
        botPanel.setFont(buttonFont);
        botPanel.setBackground(MyColors.gray204);
        Label label1 = new Label();
        Label label2 = new Label();
        Label label3 = new Label();

        Button dismissButton = new Button("Cancel");
        Button saveButton = new Button("Save Changes");
        Button helpButton = new Button("  Help  ");
        botPanel.add(label1);
        botPanel.add(dismissButton);
        botPanel.add(label2);
        botPanel.add(saveButton);
        botPanel.add(label3);
        botPanel.add(helpButton);

        Panel leftPanel = new Panel();
        leftPanel.setBackground(MyColors.gray204);
        Panel rightPanel = new Panel();
        rightPanel.setBackground(MyColors.gray204);

        this.add("South", botPanel);
        this.add("West", leftPanel);
        this.add("East", rightPanel);


        // Add inner class event handler for Dismiss button

        dismissButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
            thisHelpWindowOpen = false;
            hide();
            dispose();
            }
        });


        // Add inner class event handler for Save Changes button

        saveButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){

                int Z = IsotopePad.protonNumber;
                int N = IsotopePad.neutronNumber;
                int A = Z + N;
                double YY = 0.0;
                int nS = 0;

                // set the abundances and the color code indicating
                // that the isotope has an initial abundance

                //IsotopePad.isoColor[Z][N] = true;

                if( !IsotopePad.isAbundant[Z][N]
                    && !checkBox[2].getState()
                    && !checkBox[4].getState() ) {
                    nS = ++StochasticElements.numberSeeds;
                    IsotopePad.isAbundant[Z][N] = true;  // init abund flag
                    SegreFrame.makeRepaint();              // repaint square color
                } else {
                    nS = StochasticElements.numberSeeds;
                }

                for (int i=0; i<=4; i++) {
                    if( checkBox[i].getState() ){
                        StochasticElements.abundPref = (byte)i;
                    }
                }

                if( checkBox[0].getState() ) {
                    YY = SegreFrame.stringToDouble(massFrac.getText().trim()) / (double)A;
                    StochasticElements.initAbundMode = 1;
                } else if ( checkBox[1].getState() ) {
                    YY = SegreFrame.stringToDouble(Y.getText().trim());
                    StochasticElements.initAbundMode = 1;
                } else if ( checkBox[3].getState() || checkBox[4].getState() ) {
                    YY = SegreFrame.stringToDouble(solarAbund.getText().trim());
                    StochasticElements.initAbundMode = 3;
                }

                if( checkBox[2].getState() ) {
                    readFromFile();
                    StochasticElements.initAbundMode = 2;
                } else if ( checkBox[4].getState() ) {
                    setSolarAll();
                    StochasticElements.initAbundMode = 3;
                } else {

                    // Treat protons, He-3, & alphas different from heavy seeds

                    if (Z==1 && N==0) {
                        StochasticElements.Y[1][0]=StochasticElements.YH=YY;
                    } else if (Z==2 && N==1) {
                        StochasticElements.Y[2][1]=YY;
                    } else if (Z==2 && N==2) {
                        StochasticElements.Y[2][2]=StochasticElements.YHe=YY;
                    } else {
                        StochasticElements.seedProtonNumber[nS-1] = (byte)Z;
                        StochasticElements.seedNeutronNumber[nS-1] = (byte)N;
                        StochasticElements.seedY[nS-1] = YY;
                        StochasticElements.Y[Z][N] = YY;
                    }
                }
                hide();
                dispose();
            }
        });  // -- end inner class for Save button processing


    // Help button actions.  Handle with an inner class.

    helpButton.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ae){

            if(thisHelpWindowOpen) {
                hf.toFront();
            } else {
                hf = new GenericHelpFrame(AbundanceData.makeHelpString(),
                                            " Help for abundance setup",
                                            500,400,10,10,200,10);
                hf.show();
                thisHelpWindowOpen = true;
            }
        }
    });


    // Add window closer (inner class)

    this.addWindowListener(new WindowAdapter() {
        public void windowClosing(WindowEvent e) {
            hide();
            dispose();
        }
    });

    }  // End constructor method



    // --------------------------------------------------------------------------------------------------------
    //  Method itemStateChanged to act when state of Checkboxes changes.
    //  Requires that the class implement the
    //  ItemListener interface, which in turn requires that the
    //  method itemStateChanged be defined explicitly since
    //  ItemListener is abstract.
    // --------------------------------------------------------------------------------------------------------

    public void itemStateChanged(ItemEvent check) {

        // Process the reaction class checkboxes.  First
        // get the components of the main panel
        // and store in Component array (Note: the method
        // getComponents() is inherited from the Container
        // class by the subclass Panel).

        Component [] components0 = cboxPanel.getComponents();

        // Now process those components that are checkboxes
        // (only the odd-numbered components of the array are checkboxs).
        // First cast the Component to a Checkbox.

        Checkbox cb0 = (Checkbox)components0[1];
        Checkbox cb1 = (Checkbox)components0[3];
        Checkbox cb2 = (Checkbox)components0[5];
        Checkbox cb3 = (Checkbox)components0[7];
        Checkbox cb4 = (Checkbox)components0[9];

        // Then use the getState() method of Checkbox to
        // return boolean true if checked and false otherwise.
        // Use this logic to disable all but the chosen checkbox panel

        if( cb0.getState() ) {
        //checkBox[1].setState(false); // Seems needed despite CheckBoxGroup
        //checkBox[2].setState(false);
        //checkBox[3].setState(false);
        checkBox[0].setForeground(panelForeColor);
        checkBox[1].setForeground(disablefgColor);
        checkBox[2].setForeground(disablefgColor);
        checkBox[3].setForeground(disablefgColor);
        checkBox[4].setForeground(disablefgColor);

        massFrac.enable();
        massFrac.setBackground(textfieldColor);
        massFracL.setForeground(panelForeColor);
        Y.disable();
        Y.setBackground(disablebgColor);
        YL.setForeground(disablefgColor);
        fileName.disable();
        fileName.setBackground(disablebgColor);
        fileNameL.setForeground(disablefgColor);
        solarAbund.disable();
        solarAbund.setBackground(disablebgColor);
        solarAbundL.setForeground(disablefgColor);

        } else if ( cb1.getState() ) {
        //checkBox[0].setState(false); // Seems needed despite CheckBoxGroup
        //checkBox[2].setState(false);
        //checkBox[3].setState(false);
        checkBox[0].setForeground(disablefgColor);
        checkBox[1].setForeground(panelForeColor);
        checkBox[2].setForeground(disablefgColor);
        checkBox[3].setForeground(disablefgColor);
        checkBox[4].setForeground(disablefgColor);
        massFrac.disable();
        massFrac.setBackground(disablebgColor);
        massFracL.setForeground(disablefgColor);
        Y.enable();
        Y.setBackground(textfieldColor);
        YL.setForeground(panelForeColor);
        fileName.disable();
        fileName.setBackground(disablebgColor);
        fileNameL.setForeground(disablefgColor);
        solarAbund.disable();
        solarAbund.setBackground(disablebgColor);
        solarAbundL.setForeground(disablefgColor);

        } else if ( cb2.getState() ) {
        //checkBox[0].setState(false); // Seems needed despite CheckBoxGroup
        // checkBox[1].setState(false);
        // checkBox[3].setState(false);
        checkBox[0].setForeground(disablefgColor);
        checkBox[1].setForeground(disablefgColor);
        checkBox[2].setForeground(panelForeColor);
        checkBox[3].setForeground(disablefgColor);
        checkBox[4].setForeground(disablefgColor);
        massFrac.disable();
        massFrac.setBackground(disablebgColor);
        massFracL.setForeground(disablefgColor);
        Y.disable();
        Y.setBackground(disablebgColor);
        YL.setForeground(disablefgColor);
        fileName.enable();
        fileName.setBackground(textfieldColor);
        fileNameL.setForeground(panelForeColor);
        solarAbund.disable();
        solarAbund.setBackground(disablebgColor);
        solarAbundL.setForeground(disablefgColor);

        } else if ( cb3.getState() ) {
        //checkBox[0].setState(false); // Seems needed despite CheckBoxGroup
        //checkBox[1].setState(false);
        //checkBox[2].setState(false);
        checkBox[0].setForeground(disablefgColor);
        checkBox[1].setForeground(disablefgColor);
        checkBox[2].setForeground(disablefgColor);
        checkBox[3].setForeground(panelForeColor);
        checkBox[4].setForeground(disablefgColor);
        massFrac.disable();
        massFrac.setBackground(disablebgColor);
        massFracL.setForeground(disablefgColor);
        Y.disable();
        Y.setBackground(disablebgColor);
        YL.setForeground(disablefgColor);
        fileName.disable();
        fileName.setBackground(disablebgColor);
        fileNameL.setForeground(disablefgColor);
        solarAbund.disable();
        solarAbund.setBackground(textfieldColor);
        solarAbundL.setForeground(panelForeColor);

        } else if ( cb4.getState() ) {
        //checkBox[0].setState(false); // Seems needed despite CheckBoxGroup
        //checkBox[1].setState(false);
        //checkBox[2].setState(false);
        checkBox[0].setForeground(disablefgColor);
        checkBox[1].setForeground(disablefgColor);
        checkBox[2].setForeground(disablefgColor);
        checkBox[3].setForeground(disablefgColor);
        checkBox[4].setForeground(panelForeColor);
        massFrac.disable();
        massFrac.setBackground(disablebgColor);
        massFracL.setForeground(disablefgColor);
        Y.disable();
        Y.setBackground(disablebgColor);
        YL.setForeground(disablefgColor);
        fileName.disable();
        fileName.setBackground(disablebgColor);
        fileNameL.setForeground(disablefgColor);
        solarAbund.disable();
        solarAbund.setBackground(disablebgColor);
        solarAbundL.setForeground(disablefgColor);
        }
    }


    // -----------------------------------------------------------------------------------
    //  Method readFromFile to read abundances from a file
    // -----------------------------------------------------------------------------------

    static void readFromFile() {

        // Implement method to read abundances from file

        StochasticElements.abundFileName = fileName.getText().trim();
        try {readASCIIFile( fileName.getText().trim() );}
        catch(IOException e){ System.err.println(e.getMessage());}
        parseBuffer();
    }


    // -------------------------------------------------------------------------------------------
    //  Method setSolarAll to set solar abundances for all selected
    //  isotopes
    // -------------------------------------------------------------------------------------------

    void setSolarAll() {

        // Set solar abundances for all isotopes that are selected
        // (purple squares)

        int nS = 0;

        for(int Z=0; Z<SolarAbundances.z; Z++) {
            for(int N=0; N<SolarAbundances.n; N++) {
                if( IsotopePad.isoColor[Z][N]
                        && SolarAbundances.sab[Z][N] != 0
                        && !IsotopePad.isAbundant[Z][N]) {

                    IsotopePad.isAbundant[Z][N] = true;  // init abund flag
                    SegreFrame.makeRepaint();              // repaint this square

                    // Treat protons, He-3, & alphas different from heavy seeds

                    if (Z==1 && N==0) {
                        StochasticElements.Y[1][0] = StochasticElements.YH
                            = SolarAbundances.sab[Z][N];
                    } else if (Z==2 && N==1) {
                        StochasticElements.Y[2][1]= SolarAbundances.sab[Z][N];
                    } else if (Z==2 && N==2) {
                        StochasticElements.Y[2][2] = StochasticElements.YHe
                            = SolarAbundances.sab[Z][N];
                    } else {
                        nS = ++StochasticElements.numberSeeds;
                        StochasticElements.seedProtonNumber[nS-1] = (byte)Z;
                        StochasticElements.seedNeutronNumber[nS-1] = (byte)N;
                        StochasticElements.seedY[nS-1] = SolarAbundances.sab[Z][N];
                        StochasticElements.Y[Z][N] = SolarAbundances.sab[Z][N];
                    }
                }
            }
        }
    }


    // ----------------------------------------------------------------------------------------------
    //  The following file readin method is adapted from a program by
    //  David Flanagan.  It tests for common file input mistakes.
    // ----------------------------------------------------------------------------------------------

    public static void readASCIIFile(String from_name) throws IOException {

        File from_file = new File(from_name); //Get File objects from Strings

        // First make sure the source file exists, is a file, and is readable

        if (!from_file.exists())
            abort("no such source file: " + from_name);
        if (!from_file.isFile())
            abort("can't copy directory: " + from_name);
        if (!from_file.canRead())
            abort("source file is unreadable: " + from_name);

        // Now define a byte input stream

        FileInputStream from = null;               // Stream to read from source
        int buffLength = 32768; //16384;            // Length of input buffer in bytes

        // Copy the file, a buffer of bytes at a time.

        try {
            from = new FileInputStream(from_file); // byte input stream

            byte[] buffer = new byte[buffLength];    //Buffer for file contents
            int bytes_read;                                   //How many bytes in buffer?
            buffNumber = 0;                                 //How many buffers of length
                                                                    //buffLength have been read?

            // Read bytes into the buffer, looping until we
            // reach the end of the file (when read() returns -1).

            while((bytes_read = from.read(buffer))!= -1) {  // Read til EOF
                // Convert the input buffer to a string
                s = new String(buffer,0);
                buffNumber ++;
            }
                        
            if(buffNumber > 1){
                StochasticElements.callExit("\n*** Error in AbundanceData: Buffer size "
                    +"of buffLength="+buffLength+" exceeded for abundance read " 
                    +"\nfrom file "+from_file
                    +". Assign larger value for buffLength. ***");
            }
        }

        // Close the input stream, even if exceptions were thrown

        finally {
            if (from != null) try { from.close(); } catch (IOException e) { ; }
        }
    }


    /** A convenience method to throw an exception */
    private static void abort(String msg) throws IOException {
        throw new IOException("FileCopy: " + msg);
    }


    // -------------------------------------------------------------------------------
    //  Parse the buffer read in from the ascii file and use it
    //  to set initial abundances for isotopes
    // -------------------------------------------------------------------------------

    static void parseBuffer() {

        // Break string into tokens with whitespace delimiter

        StringTokenizer st = new StringTokenizer(s.trim());

        // Process the string tokens.  This algorithm assumes the
        // abundance data file to be in the following format:
        // Line 1 has a label of arbitrary length but there must be
        // NO WHITE SPACE IN THIS LABEL.  Line 2 has seven entries
        // separated by white space: the time to this point, the current 
        // timestep dt, the total energy released to this point, the stochastic factor, 
        // the value of massTol, the value of Ymin, and nT.  If this is a restart
        // file (input restart.out generated by previous run) these will be
        // their values at the time the restart file was written. If this is not a 
        // restart file these will all be 0.00, but THE ENTRIES MUST
        // BE THERE for the input to be parsed properly.  Then line 3
        // has 3 header labels separated by whitespace that are ignored.  
        // For each subseqent line there are three entries, separated by
        // white space: N, Z, and the molar abundance Y (note
        // that neutron number is first and proton number second).

        int nS = 0;
        int Z, N;
        double YY;
        double tempv;

        // Process and discard first line (label for the humans)

        st.nextToken();

        // Process line 2 and parse its 7 tokens into time, timestep, total energy, 
        // stochasticFactor, massTol, Ymin, and nT  (each may be zero unless this is a 
        // restart file).  Third entry will be the total energy released ERelease from the 
        // if this is a restart file. (The total energy per test particle in StochasticElements 
        // is ERelease/nT, but StochasticElements.nT is not computed yet).

        tempv = SegreFrame.stringToDouble(st.nextToken());
        if(tempv !=0) {
            StochasticElements.time = tempv;
            StochasticElements.logtmin = LOG10*Math.log(tempv);
            StochasticElements.logtminPlot = LOG10*Math.log(tempv);
        }

        tempv = SegreFrame.stringToDouble(st.nextToken());
        if(tempv !=0) StochasticElements.deltaTime = tempv;

        tempv = SegreFrame.stringToDouble(st.nextToken());
        if(tempv !=0) StochasticElements.ERelease = tempv;

        tempv = SegreFrame.stringToDouble(st.nextToken());
        if(tempv !=0) StochasticElements.stochasticFactor = tempv;

        tempv = SegreFrame.stringToDouble(st.nextToken());
        if(tempv !=0) StochasticElements.massTol = tempv;

        tempv = SegreFrame.stringToDouble(st.nextToken());
        if(tempv !=0) StochasticElements.Ymin = tempv;

        tempv = SegreFrame.stringToDouble(st.nextToken());
        if(tempv !=0) StochasticElements.nT = tempv;

        // Process and ignore next line of three headers
                
        st.nextToken();
        st.nextToken();
        st.nextToken();

        // Loop over remaining tokens and process; these now contain initial abundance
        // data (Y), either primary, or the abundances from the end of the previous
        // run if this is a restart.

        System.out.println("\nInput abundances from file "
            +StochasticElements.abundFileName+"\nwith starting ERelease="
            +StochasticElements.ERelease);

        while(st.hasMoreTokens()){

            N = Integer.parseInt(st.nextToken());
            Z = Integer.parseInt(st.nextToken());
            YY = SegreFrame.stringToDouble(st.nextToken());

            //System.out.println("Input: N="+N+" Z="+Z+" Y="+YY);

            // Following if() only allows abundance to be set if
            // the isotope has already been selected for the network
            // (already a green square) and has not had abundance
            // already set (is not an orange square).

            if( IsotopePad.isoColor[Z][N] && !IsotopePad.isAbundant[Z][N]) {

                IsotopePad.isAbundant[Z][N] = true;  // init abund flag
                SegreFrame.makeRepaint();              // repaint square color

                // Treat protons, He-3, & alphas different from heavy seeds

                if (Z==1 && N==0) {
                    StochasticElements.Y[1][0]=StochasticElements.YH=YY;
                } else if (Z==2 && N==1) {
                        StochasticElements.Y[2][1]=YY;
                } else if (Z==2 && N==2) {
                    StochasticElements.Y[2][2]=StochasticElements.YHe=YY;
                } else {
                    nS = ++StochasticElements.numberSeeds;
                    StochasticElements.seedProtonNumber[nS-1] = (byte)Z;
                    StochasticElements.seedNeutronNumber[nS-1] = (byte)N;
                    StochasticElements.seedY[nS-1] = YY;
                    StochasticElements.Y[Z][N] = YY;
                }
                if(StochasticElements.nT > 0) {
                    StochasticElements.pop[Z][N] = YY*StochasticElements.nT;
                }
            }
        }
        System.out.println();
    }


// --------------------------------------------------------------------
//  Static method to generate string for Help file
// --------------------------------------------------------------------

    static String makeHelpString() {

        String s;
        s="The parameters set through this interface control the";
        s+=" initial abundances for seeds in the network.  There are";
        s+=" five possible ways for the abundances to be entered.";
        s+=" Store them by clicking the \"Save Changes\" button.\n\n";

        s+="SPECIFY THE MASS FRACTION\nThe first option is to specify";
        s+=" the mass fraction X(i) for this isotope.\n\n";

        s+="SPECIFY THE MOLAR ABUNDANCE\nThe second option to to specify";
        s+=" the molar abundance Y(i) for this isotope.\n\n";

        s+="READ ALL ABUNDANCES FROM A FILE\nThe third option is to read all";
        s+=" abundances (for this isotope and any others with non-zero";
        s+=" abundance) from a file. The name of the file is specified by";
        s+=" the user and the format must be\n\n";
        s+=" n  z  abundance energy\n";
        s+=" 0  1  3.65E-01\n 1  2  1.9317665E-06\n 2  2  3.325E-02\n 6  6  7.8673336E-05\n\n";
        s+="where the 1st column is neutron number, the second column is proton number,";
        s+=" the 3rd column is abundances Y(i). The first line is a label that is required";
        s+=" in the format indicated (4 tokens).  The first 3 tokens are labels but the 4th is";
        s+=" the energy from a previous run if this is a restart.  In not a restart, the";
        s+=" energy entry must still be present (as 0).";
        s+=" All entries are separated by a blank space.\n\n";

        s+="CHOOSE SOLAR ABUNDANCE FOR THIS ISOTOPE\nThe fourth option is to";
        s+=" use the solar abundance for this isotope.\n\n";

        s+="CHOOSE SOLAR ABUNDANCES FOR ALL ISOTOPES\nThe final option is to";
        s+=" choose solar abundances for all isotopes (this one and all others";
        s+=" with non-zero solar abundance.";

        return s;
    }

}   /* End class AbundanceData */

