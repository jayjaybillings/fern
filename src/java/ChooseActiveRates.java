// --------------------------------------------------------------------------------------------------------
//  Class ChooseActiveRates to pop up window to allow the active rates
//  for the calculation to be specified.
// --------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;

class ChooseActiveRates extends Frame implements ItemListener {

    static boolean thisHelpWindowOpen = false;
    GenericHelpFrame hf = new GenericHelpFrame("","",0,0,0,0,0,0);

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
    final Checkbox [] checkBox = new Checkbox[3];
    Panel panel1;
    Panel cboxPanel;
    static TextField fileName;
    Label  fileNameL;
	
	int neutronSerials[] = null;
	int protonSerials[] = null;
	int deuteronSerials[] = null;
	int tritonSerials[] = null;
	int helium3Serials[] = null;
	int alphaSerials[] = null;


    // ---------------------------------------------------------------------
    //  Public constructor
    // ---------------------------------------------------------------------

    public ChooseActiveRates (int width, int height, String title, String text){

        this.pack();
        this.setSize(width,height);
        this.setTitle(title);

        // Create three checkboxes

        checkBox[0] = new Checkbox("All Rates Active");
        checkBox[1] = new Checkbox("All Rates Inactive");
        checkBox[2] = new Checkbox("Specify from file");

        // Make them part of a checkbox group (exclusive radio buttons)

        checkBox[0].setCheckboxGroup(cbg);
        checkBox[1].setCheckboxGroup(cbg);
        checkBox[2].setCheckboxGroup(cbg);

        // Add itemListeners to listen for checkbox events.  These events
        // will be processed by the method itemStateChanged

        checkBox[0].addItemListener(this);
        checkBox[1].addItemListener(this);
        checkBox[2].addItemListener(this);

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

        Label reacLabel = new Label("Choose Active Rates for Network",Label.CENTER);
        reacLabel.setFont(titleFont);
        reacLabel.setForeground(panelForeColor);
        gs.gridwidth = 2;
        cboxPanel.add(reacLabel,gs);

        // Select All button

        gs.anchor = GridBagConstraints.WEST;
        gs.gridwidth = 2;
        gs.gridy = 1;
        cboxPanel.add(checkBox[0],gs);

        // Select Individually with Mouse button

        gs.gridx = 0;
        gs.gridy = 2;
        gs.anchor = GridBagConstraints.WEST;
        cboxPanel.add(checkBox[1], gs);

        // Read from File widgets

        gs.gridx = 0;
        gs.gridy = 3;
        gs.anchor = GridBagConstraints.WEST;
        cboxPanel.add(checkBox[2], gs);

        panel1 = new Panel();
        panel1.setFont(textFont);
        panel1.setForeground(panelForeColor);

        fileNameL = new Label("File",Label.LEFT);
        panel1.add(fileNameL);

        fileName = new TextField(22);
        fileName.setFont(textFont);
        String temp = StochasticElements.activeRatesFileName;
        fileName.setText(temp);
        panel1.add(fileName);

        gs.gridx = 1;
        gs.anchor = GridBagConstraints.EAST;
        cboxPanel.add(panel1, gs);

        // Add the main container to the frame

        this.add(cboxPanel,"Center");

        // Read any values already set and place in fields

        fileName.setText(StochasticElements.activeRatesFileName);

        // Set up the initial conditions of the window

        if(StochasticElements.rateSelectionMode==1){
            checkBox[0].setState(true);
            checkBox[0].requestFocus();
            checkBox[0].setForeground(panelForeColor);
            checkBox[1].setForeground(disablefgColor);
            checkBox[2].setForeground(disablefgColor);
            fileName.disable();
            fileName.setBackground(disablebgColor);
            fileNameL.setForeground(disablefgColor);
        } else if (StochasticElements.rateSelectionMode==2){
            checkBox[1].setState(true);
            checkBox[1].requestFocus();
            checkBox[1].setForeground(panelForeColor);
            checkBox[0].setForeground(disablefgColor);
            checkBox[2].setForeground(disablefgColor);
            fileName.disable();
            fileName.setBackground(disablebgColor);
            fileNameL.setForeground(disablefgColor);
        } else {
            checkBox[2].setState(true);
            checkBox[2].setForeground(panelForeColor);
            checkBox[0].setForeground(disablefgColor);
            checkBox[1].setForeground(disablefgColor);
            fileName.setBackground(textfieldColor);
            fileName.requestFocus();
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

                // Actions to take after closing window

                if( checkBox[0].getState() ) {    // Make all rates active
                    StochasticElements.useReadRatesFlag = false;
                    StochasticElements.initialRatesZeroFlag = false;
                    StochasticElements.rateSelectionMode = 1;
                } else if ( checkBox[1].getState() ) {  // Make all rates inactive
                    StochasticElements.useReadRatesFlag = false;
                    StochasticElements.initialRatesZeroFlag = true;
                    disableAllRates();
                    hide();
                    String message = "All rates disabled. ctrl-click with the mouse to";
                    message += " add active rates for individual isotopes.";
                    ChooseIsotopes.cd.makeTheWarning(300,300,220,120,Color.black,
                        Color.lightGray, " Select with mouse", message, false,ChooseIsotopes.cd );
                    StochasticElements.rateSelectionMode = 2;
                } else if ( checkBox[2].getState() ) {  // Read which rates active from file
                    StochasticElements.useReadRatesFlag = true;
                    StochasticElements.initialRatesZeroFlag = false;
                    StochasticElements.activeRatesFileName = fileName.getText();		
                    readIncludedRates(StochasticElements.activeRatesFileName);
                    StochasticElements.rateSelectionMode = 3;
                }

                //  Create arrays in StochasticElements that hold Z and N of active isotopes.
                //  These are necessary for the truncation routines that limit reactions to those
                //  that remain in the network.
        
                StochasticElements.tabulateActiveIsotopes();

                hide();
                dispose();
            }
        });  // -- end inner class for Save button processing


    // Help button actions.  Handle with an inner class.

    helpButton.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent ae){

            String windowTitle = " Help for active rates setup";
            if(thisHelpWindowOpen) {
                hf.toFront();
            } else {
                hf = new GenericHelpFrame(ChooseActiveRates.makeHelpString(),
                    windowTitle, 500,400,10,10,200,10);
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



    // ---------------------------------------------------------------------------------------------------------
    //  Method itemStateChanged to act when state of Checkboxes changes.
    //  Requires that the class implement the
    //  ItemListener interface, which in turn requires that the
    //  method itemStateChanged be defined explicitly since
    //  ItemListener is abstract.
    // ----------------------------------------------------------------------------------------------------------

    public void itemStateChanged(ItemEvent check) {

        // Process the reaction class checkboxes.  First
        // get the components of the main panel
        // and store in Component array (Note: the method
        // getComponents() is inherited from the Container
        // class by the subclass Panel).  The components in
        // the array will be numbered in sequence from zero
        // according to the order added to the main panel.

        Component [] components0 = cboxPanel.getComponents();

        // Now process those components that are checkboxes
        // (only components 1,2,3 are checkboxes).
        // First cast the Component to a Checkbox.

        Checkbox cb0 = (Checkbox)components0[1];
        Checkbox cb1 = (Checkbox)components0[2];
        Checkbox cb2 = (Checkbox)components0[3];

        // Then use the getState() method of Checkbox to
        // return boolean true if checked and false otherwise.
        // Use this logic to disable all but the chosen checkbox panel

        if( cb0.getState() ) {
            checkBox[0].setForeground(panelForeColor);
            checkBox[1].setForeground(disablefgColor);
            checkBox[2].setForeground(disablefgColor);
            fileName.disable();
            fileName.setBackground(disablebgColor);
            fileNameL.setForeground(disablefgColor);
        } else if ( cb1.getState() ) {
            checkBox[0].setForeground(disablefgColor);
            checkBox[1].setForeground(panelForeColor);
            checkBox[2].setForeground(disablefgColor);
            fileName.disable();
            fileName.setBackground(disablebgColor);
            fileNameL.setForeground(disablefgColor);
        } else if ( cb2.getState() ) {
            checkBox[0].setForeground(disablefgColor);
            checkBox[1].setForeground(disablefgColor);
            checkBox[2].setForeground(panelForeColor);
            fileName.enable();
            fileName.setBackground(textfieldColor);
            fileNameL.setForeground(panelForeColor);
        }
    }


    // ------------------------------------------------------------------------------
    //  Method to disable all reaction rates
    // ------------------------------------------------------------------------------

    public void disableAllRates(){
        for(int Z=0; Z<110; Z++){
            for(int N=0; N<200; N++){
                for(int i=0; i<50; i++){
                    DataHolder.RnotActive[Z][N][i] = true;
                }
            }
        }
    }


    // ------------------------------------------------------------------------------
    //  Method to read in a file and parse it into variables.  Requires
    //  the class ReadAFile.
    // ------------------------------------------------------------------------------

    public void readIncludedRates(String filename) {

		for(int i=0; i<DataHolder.Znum; i++){
			for(int j=0; j<DataHolder.Nnum; j++){
				for(int k=0; k<DataHolder.reacNum; k++){
					DataHolder.RnotActive[i][j][k] = false;
				}
			}
		}
		
        //disableAllRates();
        String s = null;
		
		// Create the arrays that will allow serial index to be mapped
		// for light-light collisions if reading reactions from file.
		
		neutronSerials = findSerialArray(0,1);
		protonSerials = findSerialArray(1,0);
		deuteronSerials = findSerialArray(1,1);
		tritonSerials = findSerialArray(1,2);
		helium3Serials = findSerialArray(2,1);
		alphaSerials = findSerialArray (2, 2);

        // Create instance of ReadAFile and use it to read in the file, 
        // returning file content as a string s.

        ReadAFile raf = new ReadAFile();

        // readASCIIFile method throws IOException, so it must be caught

        try {s = raf.readASCIIFile(filename);}
        catch(IOException e){ System.err.println(e.getMessage());}

        //System.out.println("\nRaw file string:\n\n"+s+"\n");

        // ------------------------------------------------------------------------------
        //  Parse the buffer read in from the ascii file and use it
        //  to set values for variables.  
        // ------------------------------------------------------------------------------

        // Break string into tokens with whitespace delimiter
    
        StringTokenizer st = new StringTokenizer(s.trim());
    
        // Process first line:  read and ignore 3 tokens (the three column labels)
    
        st.nextToken();
        st.nextToken();
        st.nextToken();
    
        // Loop over remaining tokens and process.  
    
        int Z, N, sIndex;
		
		int reactionsRead[][] = new int[DataHolder.Znum][DataHolder.Nnum];
		
        while(st.hasMoreTokens()){
            Z = Integer.parseInt(st.nextToken());
            N = Integer.parseInt(st.nextToken());
            sIndex = Integer.parseInt(st.nextToken());
			reactionsRead[Z][N] ++;
			DataHolder.activeReactionsSerialIndex[Z][N][reactionsRead[Z][N]] = sIndex;
System.out.println("Z="+Z+" N="+N+" reactionNumber="+reactionsRead[Z][N]+" serialIndex="+sIndex);

			if(Z > 2){
				//DataHolder.useReadRates[Z][N][DataHolder.maxReadRates[Z][N]] = sIndex;
				//DataHolder.maxReadRates[Z][N] ++;
				//DataHolder.RnotActive[Z][N][sIndex] = false;
			} else  {
				//DataHolder.useReadRates[Z][N][DataHolder.maxReadRates[Z][N]] = sIndex;
				//indx = matchSerial(Z, N, sIndex);
				//DataHolder.maxReadRates[Z][N] ++;
				//DataHolder.RnotActive[Z][N][indx] = false;
			}
// if(Z==1 && N==1)System.out.println("Read-In:  Z="+Z+" N="+N+" serial="+sIndex+" indx="+indx
// 	+" DataHolder.maxReadRates="+DataHolder.maxReadRates[Z][N]
// 	+" Ractive="+!DataHolder.RnotActive[Z][N][indx]);

        }
    }


    // ------------------------------------------------------------------------------------
    //  Method to set all rates to active (if isActive = true)
    //  or non-active (if isActive = false)
    // ------------------------------------------------------------------------------------

    public void makeAllRatesActive (boolean isActive){

        System.out.println("Making all rates active = "+isActive);

        for(int Z=0; Z<=IsotopePad.zmax; Z++) {
            for(int N=0; N<=IsotopePad.nmax; N++) {
                if( SegreFrame.gp.isPStable[Z][N] ) {
                    // IsotopePad.isoColor[Z][N] = isActive;
                    // SegreFrame.gp.processSquares(Z,N);
                } 
            }
        }
    }
    
    
    // ------------------------------------------------------------------------------------
    //  Method to match a serial index to its position
    // ------------------------------------------------------------------------------------
    
    int matchSerial(int Z, int N, int serial){
		
		int index = -1;
		int temparray[] = null;
		
		if(Z==1 && N==0){
			temparray = protonSerials;
		} else if(Z==0 && N==1){
			temparray = neutronSerials;
		} else if(Z==1 && N==1){
			temparray = deuteronSerials;
		} else if(Z==1 && N==2){
			temparray = tritonSerials;
		} else if(Z==2 && N==1){
			temparray = helium3Serials;
		} else if(Z==2 && N==2){
			temparray = alphaSerials;
		} 
		
		for (int i=0; i< temparray.length; i++){
			if(serial == temparray[i]){
				return i;
			}
		}
		return -1;  // If no match
	}
	
	
	// ------------------------------------------------------------------------------------
    //  Method to read java data objects from the disk and determine the serial indices
    //  for light-light reactions.  Returns an array that has the serial indices in the
    //  order that they will be in when deserialized for the actual calculation.  Necessary
    //  to get light-light reactions to be stored properly when reactions are read from a
    //  disk file.
    // ------------------------------------------------------------------------------------
    
    int[] findSerialArray (int Z, int N){
		
		int m = 0;
		int mm = 0;
		int temparray[] = null;
		
		String file = "data/iso" + Z + "_" + N + ".ser";

        try {

            // Wrap input file stream in an object input stream

            FileInputStream fileIn = new FileInputStream(file);
            ObjectInputStream in = new ObjectInputStream(fileIn);

            // Read from the input stream the initial integer giving
            // the number of objects that were serialized in this file

            int numberObjects = in.readInt();

            // Read from the input stream the 9-member int array giving
            // the number of reactions of each type.  Entry 0 is
            // the total (=numberObjects).  Array entries 1-8 give the
            // the subtotals for each of the 8 reaction types.  We read
			// it from the stream but will not use it.

            int [] numberEachType = (int []) in.readObject();
			
			// Create a temporary array to hold the serial indices for 
			// light-light reactions involving this Z and N
			
			temparray = new int[numberObjects];
					
            while (mm < numberObjects) {
				
                ReactionClass1 tryIt = (ReactionClass1)in.readObject();
				
				// Keep only light-light reactions
				if(StochasticElements.isLightIonReaction(tryIt)){
					
					// Apply further filters to prevent double counting among the
					// light-light reactions. The same filters are used in the
					// StochasticElements method loadData() for loading light-light
					// reactions without double counting.
					
					if(Z==0 && N==1 && tryIt.reacIndex == 1
					   ||
					   Z==1 && N==0 && tryIt.reacIndex == 4 && tryIt.isoIn[1].x == 1
					   && tryIt.isoIn[1].y == 0
					   ||
					   Z==1 && N==1 &&
							(tryIt.reacIndex == 2
							|| (tryIt.reacIndex == 4 && tryIt.isoIn[1].x < 2)
							|| (tryIt.reacIndex == 5
								&& tryIt.isoIn[1].x == 1 && tryIt.isoIn[1].y == 1)
							|| tryIt.reacIndex == 6)
					   ||
					   Z==1 && N==2 &&
							(tryIt.reacIndex < 4
							|| (tryIt.reacIndex == 4 && tryIt.isoIn[1].x < 2)
							|| (tryIt.reacIndex == 5 && tryIt.isoIn[1].x < 2)
							|| tryIt.reacIndex == 6 && tryIt.isoIn[1].x < 2)
					   ||
					   Z==2 && N==1 &&
							(tryIt.reacIndex < 4
							|| (tryIt.reacIndex == 4 && tryIt.isoIn[1].y < 2)
							|| (tryIt.reacIndex == 5 && tryIt.isoIn[1].y < 2)
							|| tryIt.reacIndex == 6)
					   ||
					   Z==2 && N==2) {
							temparray[m] = mm;
							m++;
					}	   
					
				}
				mm++;
            }

            // Close the input streams
            in.close();
            fileIn.close();
        }                                        // -- end try
        catch (Exception e) {
            System.out.println(e);
        }
		
		// Return an array that is just the first m entries of the working
		// array (the rest should be zero).
		
		return Arrays.copyOfRange(temparray, 0, m);
	}
	

    // --------------------------------------------------------------------
    //  Static method to generate string for Help file
    // --------------------------------------------------------------------

    static String makeHelpString() {

        String s;
        s="This interface controls how the active rates in the";
        s+=" network are chosen. There are three choices:";
        s+="\n\n";

        s+="ALL RATES ACTIVE\nThe first option enables all rates available.\n\n";

        s+="ALL RATES INACTIVE\nIf this option is selected, after the";
        s+=" window is closed, ctrl-clicking on individual isotopes allows individual";
        s+=" rates to be set for specific isotopes.\n\n";

        s+="READ FROM A FILE\nThe third option is to read all";
        s+=" rates to be included from a file.";
        s+=" The name of the file is specified by";
        s+=" the user and the file must have the form\n\n";
        s+=" Z   N   serialIndex\n";
        s+=" 6   6   0\n 6   6   2\n 8   8   2\n 8   8   3\n\n";
        s+="where the 1st column is proton number, the second column is neutron number,";
        s+=" and the third column is the serial index for the reaction in the";
        s+=" ReactionClass1 reaction objects";
        s+=" for each isotope to be included. The first line is a label that is ignored when";
        s+=" read but must be present in the format indicated (3 tokens).";  
        s+=" Thus the preceding example will enable the reactions with serial index 0 and 2";
        s+=" for 12C and serial index 2 and 3 for 16O. If a file is read, all reactions";
        s+=" not specified explicitly in the file are disabled.";
        s+=" Entries in each line are separated by at least one blank space.\n\n";

        return s;
    }

}   /* End class ChooseActiveRates */

