
// ------------------------------------------------------------------------------------------------------------------
//  Class ChooseActiveIsotopes to pop up window to allow the active isotopes
//  for the calculation to be specified.
// ------------------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import javax.swing.*;

class ChooseActiveIsotopes extends JFrame implements ItemListener {

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

    ButtonGroup cbg = new ButtonGroup();
    final JRadioButton [] radioButton = new JRadioButton[3];
    final JComboBox networkFile = new JComboBox();
    JPanel panel1;
    JPanel cboxPanel;
    JLabel  fileNameL;


    // ---------------------------------------------------------------------
    //  Public constructor
    // ---------------------------------------------------------------------

    public ChooseActiveIsotopes (int width, int height, String title, String text){

		this.setDropTarget(null);
        this.pack();
        this.setSize(width,height);
        this.setTitle(title);
		
        // Create three JRadioButtons 
		
        radioButton[0] = new JRadioButton("Select all");
        radioButton[1] = new JRadioButton("Select isotopes with mouse");
        radioButton[2] = new JRadioButton("Specify from file");
		
		// Bug fix, see http://bugs.sun.com/bugdatabase/view_bug.do?bug_id=7027598
		// and http://jasperforge.org/plugins/espforum/view.php?group_id=102&forumid=103&topicid=83582
		radioButton[0].setDropTarget(null);
		radioButton[1].setDropTarget(null);
		radioButton[2].setDropTarget(null);
		
		networkFile.setDropTarget(null);
		
		
        
        //radioButton[0].setBackground(panelBackColor);
        radioButton[0].setFont(textFont);
        //radioButton[1].setBackground(panelBackColor);
        radioButton[1].setFont(textFont);
        //radioButton[2].setBackground(panelBackColor);
        radioButton[2].setFont(textFont);

        // Make them part of a radio button group (exclusive radio buttons)
        
        cbg.add(radioButton[0]);
        cbg.add(radioButton[1]);
        cbg.add(radioButton[2]);

        // Add itemListeners to listen for radio button events.  These events
        // will be processed by the method itemStateChanged

        radioButton[0].addItemListener(this);
        radioButton[1].addItemListener(this);
        radioButton[2].addItemListener(this);

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

        cboxPanel = new JPanel();
		cboxPanel.setDropTarget(null);
        cboxPanel.setLayout(new GridBagLayout());
        //cboxPanel.setBackground(panelBackColor);

        // Top label

        JLabel reacLabel = new JLabel("Choose Active Isotopes for Network",JLabel.CENTER);
		reacLabel.setDropTarget(null);
        reacLabel.setFont(titleFont);
        reacLabel.setForeground(panelForeColor);
        gs.gridwidth = 2;
        cboxPanel.add(reacLabel,gs);

        // Select All button

        gs.anchor = GridBagConstraints.WEST;
        gs.gridwidth = 2;
        gs.gridy = 1;
        cboxPanel.add(radioButton[0],gs);

        // Select Individually with Mouse button

        gs.gridx = 0;
        gs.gridy = 2;
        gs.anchor = GridBagConstraints.WEST;
        cboxPanel.add(radioButton[1], gs);

        // Read from File widgets

        gs.gridx = 0;
        gs.gridy = 3;
        gs.anchor = GridBagConstraints.WEST;
        cboxPanel.add(radioButton[2], gs);

        panel1 = new JPanel();
		panel1.setDropTarget(null);
        panel1.setFont(textFont);
        //panel1.setForeground(panelForeColor);
        //panel1.setBackground(panelBackColor);

        for(int i=0; i<StochasticElements.networkFileName.length; i++){
            networkFile.addItem(StochasticElements.networkFileName[i]);
        }
        networkFile.setEditable(true);
        networkFile.setFont(textFont);
        networkFile.setSelectedItem(StochasticElements.activeFileName);

        fileNameL = new JLabel("File",JLabel.LEFT);
		fileNameL.setDropTarget(null);
        fileNameL.setFont(textFont);
        panel1.add(fileNameL);
        panel1.add(networkFile);

        gs.gridx = 1;
        gs.anchor = GridBagConstraints.EAST;
        cboxPanel.add(panel1, gs);

        
		
		// Following statement gives intermittent exception.  See
		// http://bugs.sun.com/bugdatabase/view_bug.do?bug_id=7027598
		// and http://jasperforge.org/plugins/espforum/view.php?group_id=102&forumid=103&topicid=83582, 
		// but the bug fix suggested there has been tried for all J-components here and doesn't work. It seems
		// harmless, since if you click the button again it works.
		
		// Add the main container to the frame
		this.add(cboxPanel,"Center");

        // Set up the initial conditions of the window

        if(StochasticElements.isoSelectionMode==1){
            radioButton[0].setSelected(true);
            radioButton[0].requestFocus();
            radioButton[0].setForeground(panelForeColor);
            radioButton[1].setForeground(disablefgColor);
            radioButton[2].setForeground(disablefgColor);
            networkFile.disable();
            networkFile.setForeground(disablebgColor); 
            networkFile.setEditable(false);  
            fileNameL.setForeground(disablefgColor);
        } else if (StochasticElements.isoSelectionMode==2){
            radioButton[1].setSelected(true);
            radioButton[1].requestFocus();
            radioButton[1].setForeground(panelForeColor);
            radioButton[0].setForeground(disablefgColor);
            radioButton[2].setForeground(disablefgColor);
            networkFile.disable();
            networkFile.setForeground(disablebgColor); 
            networkFile.setEditable(false);  
            fileNameL.setForeground(disablefgColor);
        } else {
            radioButton[2].setSelected(true);
            radioButton[2].setForeground(panelForeColor);
            radioButton[0].setForeground(disablefgColor);
            radioButton[1].setForeground(disablefgColor);
            networkFile.enable();
            networkFile.setForeground(panelForeColor); 
            networkFile.setEditable(true);  
            networkFile.requestFocus();
        }

        // Put stuff on the bottom panel

        JPanel botPanel = new JPanel();
		botPanel.setDropTarget(null);
        botPanel.setBackground(MyColors.gray204);
        JLabel label1 = new JLabel();
        JLabel label2 = new JLabel();
        JLabel label3 = new JLabel();
		label1.setDropTarget(null);
		label2.setDropTarget(null);
		label3.setDropTarget(null);

        JButton dismissButton = new JButton("Cancel");
        JButton saveButton = new JButton("Save Changes");
        JButton helpButton = new JButton("  Help  ");
		dismissButton.setDropTarget(null);
		saveButton.setDropTarget(null);
		helpButton.setDropTarget(null);
        dismissButton.setFont(buttonFont);
        saveButton.setFont(buttonFont);
        helpButton.setFont(buttonFont);
        botPanel.add(label1);
        botPanel.add(dismissButton);
        botPanel.add(label2);
        botPanel.add(saveButton);
        botPanel.add(label3);
        botPanel.add(helpButton);

        JPanel leftPanel = new JPanel();
		leftPanel.setDropTarget(null);
        leftPanel.setBackground(MyColors.gray204);
        JPanel rightPanel = new JPanel();
		rightPanel.setDropTarget(null);
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

                if( radioButton[0].isSelected() ) {
                    makeAllIsotopesActive(true);		          // Make all isotopes active
                    StochasticElements.isoSelectionMode = 1;
                } else if ( radioButton[1].isSelected() ) {
                    //makeAllIsotopesActive(false);		  // Clear all isotopes
                    StochasticElements.isoSelectionMode = 2;
                    hide();
                    String message = "After dismissing this window, left click with the mouse to select active";
                    message += " isotopes (left click again to deselect). Then choose active rates.";
                    ChooseIsotopes.cd.makeTheWarning(300,300,240,180,Color.black,
                        MyColors.dialogColor, " Select with mouse", message, false, 
                        ChooseIsotopes.cd );
                } else if ( radioButton[2].isSelected() ) {
                    makeAllIsotopesActive (false);		// Clear all isotopes
                    StochasticElements.activeFileName = (String)networkFile.getSelectedItem();		
                    readIncludedIsotopes(StochasticElements.activeFileName);
                    StochasticElements.isoSelectionMode = 3;
                }

                hide();
                dispose();

                // Open dialog to choose active rates

                ChooseActiveRates car = new ChooseActiveRates(430,250," Choose Active Rates","text2");
                if(radioButton[1].isSelected()){
                    // If selecting isotopes with mouse, move rate selection window off to side of segre chart
                    car.move(700,300);
                } else {
                    // Otherwise center on segre chart
                    car.move(200,200);
                }
                car.show();
            }
        });  // -- end inner class for Save button processing


        // Help button actions.  Handle with an inner class.
    
        helpButton.addActionListener(new ActionListener() {
          public void actionPerformed(ActionEvent ae){

              String windowTitle = " Help for active isotopes setup";
              if(thisHelpWindowOpen) {
                  hf.toFront();
              } else {
                  hf = new GenericHelpFrame(ChooseActiveIsotopes.makeHelpString(),
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
    //  Method itemStateChanged to act when state of radio buttons changes.
    //  Requires that the class implement the
    //  ItemListener interface, which in turn requires that the
    //  method itemStateChanged be defined explicitly since
    //  ItemListener is abstract.
    // ---------------------------------------------------------------------------------------------------------

    public void itemStateChanged(ItemEvent check) {

        // Process the radio buttons.  First
        // get the components of the main panel
        // and store in Component array (Note: the method
        // getComponents() is inherited from the Container
        // class by the subclass Panel).  The components in
        // the array will be numbered in sequence from zero
        // according to the order added to the main panel.

        Component [] components0 = cboxPanel.getComponents();

        // Now process those components that are JRadioButtons
        // (only components 1,2,3 are JRadioButtons).
        // First cast the Component to a JRadioButton

        JRadioButton cb0 = (JRadioButton)components0[1];
        JRadioButton cb1 = (JRadioButton)components0[2];
        JRadioButton cb2 = (JRadioButton)components0[3];
		cb0.setDropTarget(null);
		cb1.setDropTarget(null);
		cb2.setDropTarget(null);

        // Then use the isSelected() method of JRadioButton to
        // return boolean true if checked and false otherwise.
        // Use this logic to disable all but the chosen radio button panel

        if( cb0.isSelected() ) {
            radioButton[0].setForeground(panelForeColor);
            radioButton[1].setForeground(disablefgColor);
            radioButton[2].setForeground(disablefgColor);
            networkFile.disable();
            networkFile.setForeground(disablebgColor); 
            networkFile.setEditable(false);             
            fileNameL.setForeground(disablefgColor);
        } else if ( cb1.isSelected() ) {
            radioButton[0].setForeground(disablefgColor);
            radioButton[1].setForeground(panelForeColor);
            radioButton[2].setForeground(disablefgColor);
            networkFile.disable();
            networkFile.setBackground(disablebgColor); 
            networkFile.setEditable(false); 
           fileNameL.setForeground(disablefgColor);

        } else if ( cb2.isSelected() ) {
            radioButton[0].setForeground(disablefgColor);
            radioButton[1].setForeground(disablefgColor);
            radioButton[2].setForeground(panelForeColor);
            networkFile.enable();
            networkFile.setForeground(panelForeColor); 
            networkFile.setEditable(true); 
            fileNameL.setForeground(panelForeColor);
        }
    }



    // ---------------------------------------------------------------------------------------------
    //  Method to read in a file and parse it into variables.  Requires
    //  the class ReadAFile.
    // ---------------------------------------------------------------------------------------------

    public void readIncludedIsotopes(String filename) {

        String s = null;

        // Create instance of ReadAFile and use it to read in the file, 
        // returning file content as a string s.

        ReadAFile raf = new ReadAFile(10000);

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
    
        // Process first line:  read and ignore 2 tokens (the two column labels)
    
        st.nextToken();
        st.nextToken();
    
        // Loop over remaining tokens and process.  
    
        int Z, N;
    
        while(st.hasMoreTokens()){
            Z = Integer.parseInt(st.nextToken());
            N = Integer.parseInt(st.nextToken());
            if( SegreFrame.gp.isPStable[Z][N] ) {
                IsotopePad.isoColor[Z][N] = true;
                SegreFrame.gp.processSquares(Z,N);
            }
        }
    }


    // -------------------------------------------------------------------------------------------------------------------------
    //  Method to set entire Segre plane between drip lines to active (if isActive = true)
    //  or non-active (if isActive = false) and reset colors of displayed squares
    //  accordingly.
    // -------------------------------------------------------------------------------------------------------------------------

    public void makeAllIsotopesActive (boolean isActive){

        for(int Z=0; Z<=IsotopePad.zmax; Z++) {
            for(int N=0; N<=IsotopePad.nmax; N++) {
                if( SegreFrame.gp.isPStable[Z][N] ) {
                    IsotopePad.isoColor[Z][N] = isActive;
                    SegreFrame.gp.processSquares(Z,N);
                } 
            }
        }
    }


   // --------------------------------------------------------------------
   //  Static method to generate string for Help file
   // --------------------------------------------------------------------

    static String makeHelpString() {

        String s;
        s="This interface controls how the active istopes in the";
        s+=" network are chosen. There are three choices:";
        s+="\n\n";

        s+="CHOOSE ALL ISOTOPES\nThe first option chooses all isotopes available.\n\n";

        s+="SELECT INDIVIDUALLY WITH THE MOUSE\nIf this option is selected, after the";
        s+=" window is closed, left-clicking on individual isotopes adds them (and only";
        s+=" them) to the network.\n\n";

        s+="READ FROM A FILE\nThe third option is to read all";
        s+=" isotopes to be included from a file.";
        s+=" The name of the file is specified by";
        s+=" the user and the file must have the form\n\n";
        s+=" Z   N\n";
        s+=" 6   6\n 7   7\n 8   8\n 7   6\n\n";
        s+="where the 1st column is proton number and the second column the neutron number";
        s+=" for each isotope to be included. The first line is a label that is ignored when";
        s+=" read but must be present in the format indicated (2 tokens).";  
        s+=" Thus the preceding example will select 12C, 14N, 16O, and 13N for the network.";
        s+=" Entries in each line are separated by at least one blank space.\n\n";
        s+="Besides choosing which isotopes are present in the network,";
        s+=" you may also choose to restrict which reactions involving those isotopes are";
        s+=" active in a separate choice. File options are given in a ComboBox dropdown.";

        return s;
    }

}   /* End class ChooseActiveIsotopes */

