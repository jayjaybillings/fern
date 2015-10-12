// ---------------------------------------------------------------------------------------------------
//  Class PlotParams to set up basic parameters for plotting rates.
//  Launches window in which these parameters can be specified.
//  Implements ItemListener interface to listen for Checkbox changes.
// ----------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

class PlotParams extends Frame implements ItemListener {

    static boolean helpWindowOpen = false;
    GenericHelpFrame hf = new GenericHelpFrame("","",0,0,0,0,0,0);

    Font titleFont = new java.awt.Font("SanSerif", Font.BOLD, 12);
    FontMetrics titleFontMetrics = getFontMetrics(titleFont);
    Font buttonFont = new java.awt.Font("SanSerif", Font.BOLD, 11);
    FontMetrics buttonFontMetrics = getFontMetrics(buttonFont);
    Font textFont = new java.awt.Font("SanSerif", Font.PLAIN, 12);
    FontMetrics textFontMetrics = getFontMetrics(textFont);

    Color panelForeColor = Color.black;
    Color panelBackColor = Color.white;
    Color panelBackColor2 = new Color(240,240,240);
    Color disablebgColor = new Color(230,230,230);
    Color disablefgColor = new Color(153,153,153);
    Color framebgColor = new Color(235,235,235);

    Panel panel4, panel5;
    Choice rmin, rmax, Tmin, Tmax;
    CheckboxGroup cbg = new CheckboxGroup();
    final Checkbox [] checkBox = new Checkbox[3];

    RatePlotFrame rpf = null;
    PlotReactionList rl;

    int lpoints = 80;//50;//150;
    int dpoints = 80;//80;//80;

    int x1,x2;
    double T9;


    // ---------------------------------------------------------------------------------------------------------
    //  Public constructor for PlotParams.  Takes arguments for width and
    //  height of window produced, the title for the window, and the
    //  dummy argument "text" not presently used.
    // ---------------------------------------------------------------------------------------------------------

    public PlotParams (int width, int height, String title, String text) {

        this.pack();
        this.setSize(width,height);
        this.setTitle(title);
        this.setBackground(framebgColor);

        String temp;

        Panel panel0 = new Panel();
        panel0.setLayout(new GridLayout(1,1));
        panel0.setFont(titleFont);
        panel0.setForeground(panelForeColor);

        Label first = new Label("  Reactions",Label.LEFT);
        panel0.add(first);

        // Add scrollpane window with detailed reaction list

        ScrollPane sp = new ScrollPane(ScrollPane.SCROLLBARS_AS_NEEDED);
        sp.setBackground(panelBackColor);
        sp.setForeground(panelForeColor);
        rl = new PlotReactionList();
        Panel rlist = new Panel();

        // Set a GridBagLayout for the Panel rlist and define a
        // GridBagConstraints instance that will hold the constraints
        // for components in this layout.

        rlist.setLayout(new GridBagLayout());
        GridBagConstraints constraints = new GridBagConstraints();

        // Set the relevant fields of the constraint object
        constraints.weightx = constraints.weighty = 100.0;  // Relative weights
        constraints.gridx = 0;                      // x position for component
        constraints.gridy = 0;                      // y position for component
        constraints.ipadx = 0;                     // Internal padding for component in x
        constraints.ipady = 0;                     // Internal padding for component in y
        constraints.anchor = GridBagConstraints.NORTHWEST;  // Anchor upper left
        constraints.insets = new Insets(5,5,5,5);  // Margins
        rlist.add(rl,constraints);   // Add ReactionList rl to Panel rlist,
                                            // constrained by constraint object.
        sp.add(rlist);                  // Add Panel rlist to ScrollPane sp

        Panel panel2 = new Panel();
        panel2.setLayout(new FlowLayout());
        panel2.setFont(textFont);
        panel2.setForeground(panelForeColor);

        Label rminL = new Label("log Rmin",Label.RIGHT);
        panel2.add(rminL);

        rmin = new Choice();
        rmin.setFont(textFont);
        rmin.setBackground(panelBackColor);
        for(int i=10; i>-25; i--) { rmin.addItem(Integer.toString(i));}
        rmin.select(30);
        panel2.add(rmin);

        Label rmaxL = new Label("log Rmax",Label.RIGHT);
        panel2.add(rmaxL);

        rmax = new Choice();
        rmax.setFont(textFont);
        rmax.setBackground(panelBackColor);
        for(int i=25; i>-21; i--) { rmax.addItem(Integer.toString(i));}
        rmax.select(15);
        panel2.add(rmax);

        Panel panel2B = new Panel();
        panel2B.setLayout(new FlowLayout());
        panel2B.setFont(textFont);
        panel2B.setForeground(panelForeColor);

        Label TminL = new Label("log Tmin",Label.RIGHT);
        panel2B.add(TminL);

        Tmin = new Choice();
        Tmin.setFont(textFont);
        Tmin.setBackground(panelBackColor);
        for(int i=5; i<15; i++) { Tmin.addItem(Integer.toString(i)); }
        Tmin.select(2);
        panel2B.add(Tmin);

        Label TmaxL = new Label("log Tmax",Label.RIGHT);
        panel2B.add(TmaxL);

        Tmax = new Choice();
        Tmax.setFont(textFont);
        Tmax.setBackground(panelBackColor);

        for(int i=5; i<15; i++) { Tmax.addItem(Integer.toString(i)); }
        Tmax.select(5);
        panel2B.add(Tmax);

        // Create three checkboxes

        checkBox[0] = new Checkbox("log-log");
        checkBox[1] = new Checkbox("lin-log");
        checkBox[2] = new Checkbox("lin-lin");

        // Make them part of a checkbox group (exclusive radio buttons)

        checkBox[0].setCheckboxGroup(cbg);
        checkBox[1].setCheckboxGroup(cbg);
        checkBox[2].setCheckboxGroup(cbg);

        // Set the first checkbox true

        cbg.setSelectedCheckbox(checkBox[0]);

        // Add itemListeners to listen for checkbox events.  These events
        // will be processed by the method itemStateChanged

        checkBox[0].addItemListener(this);
        checkBox[1].addItemListener(this);
        checkBox[2].addItemListener(this);

        panel4 = new Panel();
        panel4.setLayout(new FlowLayout());
        panel4.setFont(textFont);
        panel4.setForeground(panelForeColor);

        panel4.add(checkBox[0]);
        panel4.add(checkBox[1]);
        panel4.add(checkBox[2]);

        // Panel to hold all the texfields, labels, and checkboxes.
        // Lay out with GridBagLayout.

        Panel cboxPanel = new Panel();
        cboxPanel.setLayout(new GridBagLayout());
        GridBagConstraints cs = new GridBagConstraints();
        cs.weightx = 100;
        cs.weighty = 100;
        cs.fill=GridBagConstraints.BOTH;
        cs.gridx = 0;
        cs.gridy = 0;
        cs.ipadx = 0;
        cs.ipady = 0;
        cs.gridwidth=1;
        cs.gridheight=1;
        cs.anchor = GridBagConstraints.NORTH;
        cs.insets = new Insets(7,7,7,7);
        cboxPanel.add(sp,cs);

        cs.gridy = 1;
        cs.fill=GridBagConstraints.NONE;
        cs.weighty = 2;
        cs.insets = new Insets(0,0,0,10);
        cboxPanel.add(panel2, cs);

        cs.gridy = 2;
        cs.weighty = 2;
        cboxPanel.add(panel2B, cs);

        cs.gridy = 3;
        cs.weighty = 2;
        cs.insets = new Insets(0,10,0,0);
        cboxPanel.add(panel4, cs);

        // Add the cboxPanel to Panel p

        this.add(cboxPanel,"Center");

        // Add Dismiss, Save Changes, Print, and Help buttons

        Panel botPanel = new Panel();
        botPanel.setFont(buttonFont);
        botPanel.setBackground(MyColors.gray204);

        Button dismissButton = new Button("Cancel");
        Button plotButton = new Button("Plot");
        Button printButton = new Button("Print");
        Button helpButton = new Button("Help");
        botPanel.add(dismissButton);
        botPanel.add(plotButton);
        botPanel.add(printButton);
        botPanel.add(helpButton);

        this.add("South", botPanel);


        // Add inner class event handler for Dismiss button

        dismissButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
              if (PlotParams.this.rpf != null) {
                  PlotParams.this.rpf.hide();
                  PlotParams.this.rpf.dispose();
              }
              hide();
              dispose();
            }
        });


        // Add inner class event handler for plot button

        plotButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){

            try {       // Catch NumberFormatExceptions and warn user

                if(rpf == null) {       // Create if doesn't exist
                    doRatePlotFrame();
                } else {                // Update if it exists
                    rpfInitialize();
                    rpf.hide();
                    rpf.show();
                }

                //hide();
                //dispose();

            }
            catch(NumberFormatException e) {
                String screwup = "NumberFormatException.";
                screwup += "  At least one required data field has";
                screwup += " a blank or invalid entry.";
                System.out.println(screwup);
                //MyWarning np = new MyWarning(300,300,200,150,Color.black,
                //                 Color.lightGray, " Warning!",
                //                 screwup, false );
                //  np.show();

                makeTheWarning(300,300,200,150,Color.black,
                                 Color.lightGray, " Warning!",
                                 screwup, false, PlotParams.this);
            }


            }
        });  // -- end inner class for Save button processing


        // Help button actions.  Handle with an inner class.
        
        helpButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){

                if(helpWindowOpen) {
                    hf.toFront();
                } else {
                    hf = new GenericHelpFrame(PlotParams.makeHelpString(),
                        " Help for parameter setup", 500,400,10,10,200,10);
                    hf.show();
                    helpWindowOpen = true;
                }
            }
        });


        // Add inner class event handler for Print button

        printButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
                printThisFrame(20,20,false);
            }
        });


        // Add window closing button (inner class)

        this.addWindowListener(new WindowAdapter() {
           public void windowClosing(WindowEvent e) {
              if(PlotParams.this.rpf != null) {
                  PlotParams.this.rpf.hide();
                  PlotParams.this.rpf.dispose();
                  StochasticElements.amPlottingRates = false;
              }
              hide();
              dispose();
           }
        });

    }  /* End contructor */




    // ------------------------------------------------------------------------------------------------------------
    //  Method itemStateChanged to act when state of Checkboxes changes.
    //  Requires that the class implement the
    //  ItemListener interface, which in turn requires that the
    //  method itemStateChanged be defined explicitly since
    //  ItemListener is abstract.
    // ------------------------------------------------------------------------------------------------------------

    public void itemStateChanged(ItemEvent check) {
/*
        // Process the reaction class checkboxes.  First
        // get the components of the relevant panels
        // and store in Component arrays (Note: the method
        // getComponents() is inherited from the Container
        // class by the subclass Panel).

        Component [] components4 = panel4.getComponents();
        Component [] components5 = panel5.getComponents();

        // Now process these components that are checkboxes
        // (only the first element of each array is).  First cast the
        // Component to a Checkbox.  Then use the getState()
        // method of Checkbox to return boolean true if
        // checked and false otherwise.

        Checkbox cb4 = (Checkbox)components4[0];  // Checkbox for panel4
        Checkbox cb5 = (Checkbox)components5[0];  // Checkbox for panel5

        // Then use the getState() method of Checkbox to
        // return boolean true if checked and false otherwise.
        // Use this logic to disable one or the other sets of
        // choices for temperature and density input.

        if( cb4.getState() ) {
           checkBox[1].setState(false); // Seems needed despite CheckBoxGroup
           rho.disable();
           rho.setBackground(disablebgColor);
           rhoL.setForeground(disablefgColor);
           T9.disable();
           T9.setBackground(disablebgColor);
           T9L.setForeground(disablefgColor);
           profile.enable();
           profile.setBackground(panelBackColor);
           profileL.setForeground(panelForeColor);
        } else if ( cb5.getState() ) {
           checkBox[0].setState(false);
           rho.enable();
           rho.setBackground(panelBackColor);
           rhoL.setForeground(panelForeColor);
           T9.enable();
           T9.setBackground(panelBackColor);
           T9L.setForeground(panelForeColor);
           profile.disable();
           profile.setBackground(disablebgColor);
           profileL.setForeground(disablefgColor);
       }
  */

    }




    // -------------------------------------------------------------------------------------------------
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
    // --------------------------------------------------------------------------------------------------

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



    // --------------------------------------------------------------------------------------
    //  Method doRatePlotFrame to create a plot frame for rates
    // --------------------------------------------------------------------------------------

    public void doRatePlotFrame() {

        if( rl.numberGroups + rl.numberComponents > rpf.gp.imax ) {
            String tmp = "Too many reactions (";
            tmp += (rl.numberGroups+rl.numberComponents)+").  ";
            tmp += "Maximum number of curves = "+rpf.gp.imax;
            tmp += " (total rates + component rates).  Close Plotting";
            tmp += " Parameters window, return to";
            tmp += " original isotope/reaction selection window,";
            tmp += " and reduce the number of reactions and components.";

            makeTheWarning(300,300,280,170,
                  Color.black, MyColors.warnColorBG, " Error!", tmp,
                  false, PlotParams.this);

            return;  // Break out of both for loops
        }


        StochasticElements.amPlottingRates = true;
        
        // Create a customized plot frame and display it
        rpf = new RatePlotFrame();

        rpfInitialize();
        dataFiller();
        rpfInitialize();

        //  Create a menu bar and add menu to it for the plot frame
        MenuBar plotmb = new MenuBar();
        rpf.setMenuBar(plotmb);
        Menu plotMenu = new Menu("File");
        plotmb.add(plotMenu);

        //  Create menu items with keyboard shortcuts for the plot frame
        MenuItem ss,pp,qq;
        plotMenu.add(ss=new MenuItem("Save as Postscript",
                                             new MenuShortcut(KeyEvent.VK_S)));
        plotMenu.add(pp=new MenuItem("Print", new MenuShortcut(KeyEvent.VK_P)));
        plotMenu.addSeparator();     //  Menu separator
        plotMenu.add(qq=new MenuItem("Quit", new MenuShortcut(KeyEvent.VK_Q)));

        // Create and register action listeners for menu items
        ss.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e){
                RatePlotFileDialogue fd =
                  new RatePlotFileDialogue(100,100,250,110,Color.black,
                    Color.lightGray,"Choose File Name",
                    "Choose a postscript file name:");
                fd.setResizable(false);
                fd.show();
            }
        });

        pp.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e){
                rpf.printThisFrame(55,70,false);}
        });

        qq.addActionListener(new ActionListener() {
          public void actionPerformed(ActionEvent e){
            rpf.hide();
            rpf.dispose();
          }
        });

        rpf.show();

    }



    // -----------------------------------------------------------------------------------
    //  Method rpfInitialize() to initialize the parameters of
    //  RatePlotFrame instance rpf.
    // -----------------------------------------------------------------------------------

    public void rpfInitialize() {

        rpf.pack();
        rpf.setSize(510,580);
        rpf.setTitle(" Thermonuclear Reaction Rates");
        rpf.setLocation(20,50);
        rpf.setResizable(false);
        rpf.setBackground(new Color (220,220,220));
        rpf.gp.xtitle = "Temperature (K)";
        rpf.gp.ytitle = "Rate";
        x1 = SegreFrame.stringToInt(rmin.getSelectedItem());
        rpf.gp.ymin = Math.pow(10,x1);
        x2 = SegreFrame.stringToInt(rmax.getSelectedItem());
        rpf.gp.ymax = Math.pow(10,x2);
        rpf.gp.ytickIntervals = Math.abs(x1-x2);
        x1 = SegreFrame.stringToInt(Tmin.getSelectedItem());
        rpf.gp.xmin = Math.pow(10,x1);
        x2 = SegreFrame.stringToInt(Tmax.getSelectedItem());
        rpf.gp.xmax = Math.pow(10,x2);
        rpf.gp.xtickIntervals = Math.abs(x1-x2);
        if( checkBox[0].getState() ) {             // log-log
            rpf.gp.plotmode = 2;
            rpf.gp.xdplace = rpf.gp.ydplace = 0;
        } else if ( checkBox[1].getState() ) {     // log-lin
            rpf.gp.plotmode = 1;
            rpf.gp.xdplace = 4;
            rpf.gp.ydplace = 0;
        } else if ( checkBox[2].getState() ) {     // lin-lin
            rpf.gp.plotmode = 0;
            rpf.gp.xdplace = rpf.gp.ydplace = 4;
        }

        //dataFiller();

        // Set show/noshow flags for total curves

        for (int i=0; i<rl.numberGroups; i++) {
            if( rl.cbt[i].getState() ) { rpf.gp.doplot[i] = 1; }
            else {rpf.gp.doplot[i] = 0; }
        }

        // Set show/noshow flags for component curves

        for (int i=0; i<rl.numberComponents; i++) {
            System.out.println("i="+i+" cblength="+rl.cb.length+" cbstate=" +rl.cb[i].getState());
            if( rl.cb[i].getState() ) { rpf.gp.doplot[i+rl.numberGroups] = 1; }
            else {rpf.gp.doplot[i+rl.numberGroups] = 0; }
        }

    }


    // ----------------------------------------------------------------------------
    //  Method dataFiller() to fill plotting arrays with rates
    // ----------------------------------------------------------------------------

    void dataFiller() {

        int mainIndex = 0;
        int numberFound = 0;

        // Total rate curves

        for (int i=0; i<rl.numberGroups; i++) {
            loadTotalCurve(mainIndex,numberFound);
            numberFound ++;
            if(numberFound == rl.numberGroups) { break; }
            mainIndex += rl.reactionGroups[numberFound-1];
        }

        // Component rate curves

        mainIndex = 0;
        for (int i=0; i<rl.numberGroups; i++) {
            if(rl.reactionGroups[i] > 1) {
                for (int j=0; j<rl.reactionGroups[i]; j++) {
                    System.out.println("mainIndex="+mainIndex+" reaction = "+rl.rArray[mainIndex].reacString);
                    loadPartialCurve(mainIndex,numberFound);
                    numberFound ++;
                    if(numberFound == rl.numberGroups) { break; }
                    mainIndex ++;
                }
            } else { mainIndex ++; }
        }

    }


    // ------------------------------------------------------------------------------------------------
    //  Method loadTotalCurve to load data for the total reaction rates
    // ------------------------------------------------------------------------------------------------

    void loadTotalCurve(int mainIndex, int numberFound) {

        double dlogt = (double)(x2-x1)/(double)lpoints;
        double dt = (Math.pow(10,x2) - Math.pow(10,x1))/(double)lpoints;
        for (int i=0; i<lpoints; i++) {
            if( checkBox[0].getState() ) {
                T9 = Math.pow(10,(double)x1 + (double)i*dlogt)/1.0E9;
            } else {
                T9 = (Math.pow(10,x1) + (double)i*dt)/1.0E9;
            }

            // With new faster rate calculation with powers and logs of T computed once per
            // timestep, need to specify these quantities in ReactionClass1 before calling the
            // .rate method (As of svn revision 397, Feb. 18, 2009).

            ReactionClass1.logT9 = Math.log(T9);
            ReactionClass1.T913 = Math.pow(T9,0.3333333);
            ReactionClass1.T953 = Math.pow(T9,1.6666666);

            double temp = 0.0;
            for (int j=0; j<rl.reactionGroups[numberFound]; j++) {
                temp += rl.rArray[mainIndex+j].rate(T9);
            }

            // Prevent underflow or overflow which would cause problems with log-log plots

            temp = Math.min(temp, 1.0E100);
            temp = Math.max(temp, 1.0E-100);
            rpf.gp.y[numberFound][i] = temp;
            rpf.gp.x[numberFound][i] = T9*1.0E9;

        }
        rpf.gp.curvetitle[numberFound] = rl.rArray[mainIndex].reacString;
        rpf.gp.npoints[numberFound] = lpoints;
        rpf.gp.numberCurves = numberFound+1;
        rpf.gp.mode[numberFound] = 1;     // plot as lines
        rpf.gp.doplot[numberFound] = 1;    // make plot visible initially
    }



    // ---------------------------------------------------------------------------------------
    //  Method loadPartialCurve to load data for the partial rates
    // ---------------------------------------------------------------------------------------

    void loadPartialCurve(int mainIndex, int numberFound) {

        double dlogt = (double)(x2-x1)/(double)dpoints;
        double dt = (Math.pow(10,x2) - Math.pow(10,x1))/(double)dpoints;
        for (int i=0; i<dpoints; i++) {
            if( checkBox[0].getState() ) {
                T9 = Math.pow(10,(double)x1 + (double)i*dlogt)/1.0E9;
            } else {
                T9 = (Math.pow(10,x1) + (double)i*dt)/1.0E9;
            }

            // With new faster rate calculation with powers and logs of T computed once per
            // timestep, need to specify these quantities in ReactionClass1 before calling the
            // .rate method (As of svn revision 397, Feb. 18, 2009).

            ReactionClass1.logT9 = Math.log(T9);
            ReactionClass1.T913 = Math.pow(T9,0.3333333);
            ReactionClass1.T953 = Math.pow(T9,1.6666666);

            double temp = 0.0;
            temp = rl.rArray[mainIndex].rate(T9);

            // Prevent underflow or overflow which would cause problems with log-log plots

            temp = Math.min(temp, 1.0E100);
            temp = Math.max(temp, 1.0E-100);
            rpf.gp.y[numberFound][i] = temp;
            rpf.gp.x[numberFound][i] = T9*1.0E9;

        }

        String ts = "";
        if( rl.rArray[mainIndex].resonant ) {ts = " (r)"; }
        else { ts = " (nr)"; }
        rpf.gp.curvetitle[numberFound] = rl.rArray[mainIndex].reacString + ts;
        rpf.gp.doplot[numberFound] = 0;
        rpf.gp.npoints[numberFound] = dpoints;
        rpf.gp.numberCurves = numberFound+1;
        rpf.gp.mode[numberFound] = 8;
    }




    // ---------------------------------------------------------------------------------------------------------------------
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




    // ---------------------------------------------------------------------------------------------------
    //  Static method makeHelpString() to generate string for Help file
    // ---------------------------------------------------------------------------------------------------

    static String makeHelpString() {

        String s;
        s="The reactions to be plotted (and their components) may be selected or";
        s+=" deselected using the check boxes. The parameters set through this";
        s+=" interface control the details";
        s+=" of rate plotting.  Once they are set, initiate a rates plot";
        s+=" by clicking the \"Plot\" button.\n\n";

        s+="log Rmin\nBase-10 logarithm of the minimum for the rate axis";
        s+=" plot, in inverse seconds.\n\n";

        s+="log Rmax\nBase-10 logarithm of the maximum for the rate axis,";
        s+=" plot.  Units depend on the reaction. One-body rates (decays,";
        s+=" photodisintegrations) are in inverse seconds;";
        s+=" 2-body and 3-body rates must be multiplied by density and abundance factors";
        s+=" to be converted to inverse seconds.\n\n";

        s+="log Tmin\nBase-10 logarithm of the minimum for the temperature";
        s+=" axis, in Kelvin.\n\n";

        s+="log Tmax\nBase-10 logarithm of the maximum for the temperature";
        s+=" axis, in Kelvin.\n\n";

        s+="log-log / log-lin / lin-lin\nChoose log-log, log-linear, or";
        s+=" linear-linear for the plots.\n\n";

        return s;
    }

}   /* End class PlotParams */

