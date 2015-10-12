// -----------------------------------------------------------------------------------------------
//  Class PlotReactionList generates a panel with reactions whose
//  rates are to be plotted listed in a scrollable field.
// -----------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

class PlotReactionList extends Panel {

    Color color1 = new Color(210,210,210);
    Color color2 = new Color(235,235,235);

    int maxCases = 50;     // Max number of reaction entries
    int nR;                        // Actual number of reactions
    Checkbox [] cbt = new Checkbox[maxCases];
    Checkbox [] cb = new Checkbox[maxCases];
    ReactionClass1 [] rArray = new ReactionClass1[maxCases];
    int [] reactionGroups = new int[maxCases];

    int numberGroups;             // Number of groups of reaction components

    int numberComponents;     // Total number of components in groups
                                            // with more than one component.  The sum
                                            // of numberGroups and numberComponents is
                                            // the total number of curves passed to
                                            // the plotting program.


    // -------------------------------------------------------------
    //  Public constructor for PlotReactionList
    // -------------------------------------------------------------

    public PlotReactionList() {

        nR = loadData();

        // Create nR checkboxes with reaction labels.  To get a scrolling
        // list of checkboxes for total reactions and indented checkboxes
        // for the components of those reactions, we will use a combination
        // of a vertical GridLayout of panels containing the checkboxes and
        // their labels, and a GridBagLayout on each panel to place the
        // checkbox with the desired indentation.

        reactionGrouper();  // To determine how components are grouped

        // Layout for the vertical set of panels holding the checkboxes

        this.setLayout(new GridLayout(nR+numberGroups,1,5,5));

        // The set of panels that will hold the checkboxes

        Panel [] cbHolder = new Panel [nR + numberGroups];

        // Define a GridBagConstaints object that will hold the constraints
        // for the later GridBagLayout of the checkboxes on each panel

        GridBagConstraints cs = new GridBagConstraints();

        // Set the initial relevant fields of the constraint object

        cs.weightx = cs.weighty = 100.0;        // Relative weights
        cs.gridx = 0;                                     // x position for component
        cs.gridy = -1;                                    // y position for component
        cs.ipadx = 0;                                     // Internal padding for component in x
        cs.ipady = 0;                                     // Internal padding for component in y
        cs.anchor = GridBagConstraints.WEST;    // Anchor left
        cs.insets = new Insets(0,5,0,0);                 // Margins

        int placeCounter = 0;
        int gplaceCounter = 0;
        int countSingles = 0;
        int panelCounter = -1;

        for (int k=0; k<numberGroups; k++) {
            panelCounter ++;
            cbt[k] = new Checkbox( rArray[placeCounter].reacString, true );
            cbHolder[panelCounter] = new Panel();
            cbHolder[panelCounter].setLayout(new GridBagLayout());
            cbHolder[panelCounter].setBackground(color1);
            cs.insets=new Insets(0,4,0,0);                  // Inset from left margin
            cbHolder[panelCounter].add( cbt[k], cs );    // Add checkbox + constraints
            this.add(cbHolder[panelCounter]);              // Add panel

            if(reactionGroups[k] != 1) {
                cs.insets=new Insets(0,24,0,0);         // Inset from left margin
                for (int i=gplaceCounter; i<gplaceCounter+reactionGroups[k]; i++) {
                    String cbString = stringSetter(i);
                    panelCounter ++;
                    int indy=i;
                    if (indy != 0) indy = i-countSingles;
                    cbHolder[panelCounter] = new Panel();
                    cbHolder[panelCounter].setLayout(new GridBagLayout());
                    cbHolder[panelCounter].setBackground(color2);
                    cb[indy] = new Checkbox(""+cbString,false);
                    cbHolder[panelCounter].add(cb[indy], cs);      // Add with constraints
                    this.add(cbHolder[panelCounter]);                 // Add panel
                }
            } else {countSingles++;}

            gplaceCounter += reactionGroups[k];
            placeCounter += reactionGroups[k];
        }
    }



    // ----------------------------------------------------------------------------------------
    //  Method reactionGrouper() to sort through reaction list
    //  and group components of the same reaction.  In the
    //  reaction library each reaction is represented by from
    //  1 to 5 separate entries: a non-resonant component
    //  and 0-4 resonant components.  This method creates
    //  an array holding the number of components for each group
    //  of reactions in the reaction list rArray[].  It does so
    //  by exploiting the fact that the Q value is the same
    //  for all components of the same reaction.  For example,
    //  if the first entry in rArray[] is the nonresonant
    //  component of a reaction and the next two entries are
    //  resonant components for the same reaction, the array
    //  reactionGroups[] will hold the integer 3 in its first
    //  position (index 0), while the first three entries of the
    //  array rArray[] will hold the reaction objects for these
    //  three components.  The information in these two arrays
    //  then allows one to choose whether to plot the total rate
    //  and/or components of that rate.
    // -----------------------------------------------------------------------------------------

    public void reactionGrouper() {

        int i=0;
        while (i < nR) {
            int gCounter = 1;
            if (i < nR-1 && rArray[i].Q == rArray[i+1].Q) { gCounter ++; }
            if (i < nR-2 && rArray[i].Q == rArray[i+2].Q) { gCounter ++; }
            if (i < nR-3 && rArray[i].Q == rArray[i+3].Q) { gCounter ++; }
            if (i < nR-4 && rArray[i].Q == rArray[i+4].Q) { gCounter ++; }
            if (i < nR-5 && rArray[i].Q == rArray[i+5].Q) { gCounter ++; }

            reactionGroups[numberGroups] = gCounter;
            numberGroups ++;
            i += gCounter;
        }

        // Count the number of components that will be passed to
        // the plotting program.  Only components in groups with
        // more than one component are counted (since in a 1-component
        // group the component is already plotted as the total curve).
        // The total number of curves passed to the plotting program
        // is numberGroups + numberComponents.

        for (i=0; i<numberGroups; i++) {
            if(reactionGroups[i] != 1) {
                numberComponents += reactionGroups[i];
            }
        }
    }



    // ----------------------------------------------------------------------------------------
    //  Method stringSetter to construct the string describing the
    //  reaction labeled by the index i.
    // ----------------------------------------------------------------------------------------

    public String stringSetter(int i) {

        String cb = rArray[i].reacString;
        if( rArray[i].nonResonant ) { cb += "  (nr)"; }
        else if( rArray[i].resonant ) { cb += "  (r)"; }
        return cb;

    }


    // -------------------------------------------------------------------------------------------
    //  Method loadData() to load data into reaction array rArray[]
    //  by deserializing reaction objects from disk.  Returns
    //  the length of the array rArray[] after deserialization
    //  is complete.
    // -------------------------------------------------------------------------------------------

    public int loadData() {

        int m = 0;
        int mm = 0;

        // Loop over Z and N and pick up any selected reactions
        // for isotopes that have been selected

        breakLabel:       // Label for labeled break

        for (int Z=0; Z<StochasticElements.pmax; Z++) {
            for (int N=0; N<StochasticElements.nmax; N++) {
                if ( IsotopePad.isoColor[Z][N] ) {

                    mm = 0;

                    // Construct name of the serialized file corresponding
                    // to this isotope.  These files are produced
                    // automatically by the class FriedelParser from the
                    // Thielemann reaction library.  They should be in a
                    // subdirectory of the present directory called "data",
                    // and their names have the standard form "isoZ_N.ser",
                    // where Z is the proton number and N the neutron number
                    // of the isotope.

                    String file = "data/iso" + Z + "_" + N + ".ser";

                    try {

                        // Wrap input file stream in an object input stream

                        FileInputStream fileIn = new FileInputStream(file);
                        ObjectInputStream in = new ObjectInputStream(fileIn);

                        // Read from the input stream the initial integer giving
                        // the number of objects that were serialized in this file.

                        int numberObjects = in.readInt();

                        // Read from the input stream the 9-member int array giving
                        // the number of reactions of each type.  Entry 0 is
                        // the total (=numberObjects).  Array entries 1-8 give the
                        // the subtotals for each of the 8 reaction types (read
                        // it but we won't use it here).

                        int [] numberEachType = (int []) in.readObject();

                        // Deserialize the reaction objects to the array rArray []

                        while (mm < numberObjects) {
                            ReactionClass1 tryIt = (ReactionClass1)in.readObject();
                            System.out.println("file="+file+" length="+numberObjects
                                +" label="+tryIt.reacString
                                +" RnotActive="+DataHolder.RnotActive[Z][N][m]+" mm="+mm+" no="+numberObjects);

                            if( !DataHolder.wasOpened[Z][N] ) {
                                DataHolder.RnotActive[Z][N][m] =
                                    !SegreFrame.includeReaction[tryIt.reacIndex];
                            }
                            
                            mm++;       // Here rather than below to keep the continue statements from causing eof read error

                            if( !DataHolder.RnotActive[Z][N][m] ) {
                                if(Z==0 && N==1 && !StochasticElements.isLightIonReaction(tryIt)) continue;
                                if(Z==1 && N==0 && !StochasticElements.isLightIonReaction(tryIt)) continue;
                                if(Z==2 && N==2 && !StochasticElements.isLightIonReaction(tryIt)) continue;
                                rArray[m] = tryIt;
                                if(m < maxCases -1) { m++; }
                                else {

                                    // Note that the final argument for
                                    // makeTheWarning below must be a
                                    // reference to a Frame.  The present
                                    // object is not a Frame, so we reference
                                    // the Frame in which it sits (The instance
                                    // SegreFrame.pm of PlotParams).

                                    makeTheWarning(300,300,300,100,
                                        Color.black, MyColors.warnColorBG, " Warning!",
                                        "Too many reactions! Truncating list.",
                                        true, SegreFrame.pm);

                                    break breakLabel;  // Break out of both for loops
                                }
                            }

                            //mm ++;
                        }

                        // Close the input streams

                        in.close();
                        fileIn.close();
                    }                                 // -- end try
                    catch (Exception e) {
                        System.out.println(e);
                    }

                }        // if
            }            // N
        }                // Z

        return m;        // Number of reactions stored

    }




    // -----------------------------------------------------------------------------------------------------------------------
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
    // ----------------------------------------------------------------------------------------------------------------------


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

}      /* End class PlotReactionList */

