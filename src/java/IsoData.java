// ---------------------------------------------------------------------------------------------------------
// Class IsoData to pop up window to examine reaction list for a specific
// isotope (triggered by ctrl-press on isotope square)
// ---------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;


class IsoData extends Frame {

    Font titleFont = new java.awt.Font("SanSerif", Font.BOLD, 12);
    FontMetrics titleFontMetrics = getFontMetrics(titleFont);
    Font buttonFont = new java.awt.Font("SanSerif", Font.BOLD, 11);
    FontMetrics buttonFontMetrics = getFontMetrics(buttonFont);

    Color panelForeColor = Color.black;
    Color panelBackColor = new Color(204,204,204);
    Color panelBackColor2 = new Color(240,240,240);
    Color panelBackColor3 = new Color(190,190,190);

    ReactionList rl;

    static Checkbox [] checkBox = new Checkbox[9];

    int Z = IsotopePad.protonNumber;
    int N = IsotopePad.neutronNumber;

    int lenny;

    // ----------------------------------------------------------------
    //  Public constructor
    // ----------------------------------------------------------------

    public IsoData (int width, int height, String title, String text){
        
        String nn = "Z = " + Z;
        nn += "    N = " + N;
        String mass = String.valueOf(Z+N);

        this.pack();
        this.setSize(width,height);
        this.setTitle(title);

        Panel tPanel = new Panel();
        Panel tPanel1 = new Panel();
        Panel tPanel2 = new Panel();
        tPanel.setFont(titleFont);
        tPanel.setLayout(new GridLayout(1,1,5,5));

        tPanel.setForeground(panelForeColor);
        tPanel.setBackground(panelBackColor);
        Label reacLabel = new Label (nn, Label.CENTER);
        reacLabel.setBackground(panelBackColor3);
        tPanel.add(reacLabel);

        // Panel to hold top label and checkboxes

        final Panel cboxPanel = new Panel();     // Must be final because
                                                                    // used in later inner
                                                                    // class button event handler

        cboxPanel.setLayout(new GridLayout(9,1,5,5));
        cboxPanel.setForeground(panelForeColor);
        cboxPanel.setBackground(panelBackColor);

        // Create checkboxes

        for(int i=1; i<=8; i++) {
            checkBox[i] = new Checkbox(" class " + String.valueOf(i));
        }

        // Add top label and checkboxes to the cboxPanel panel

        cboxPanel.add(tPanel);

        for(int i=1; i<=8; i++) {
            cboxPanel.add(checkBox[i]);
        }

        // Add the checkbox panel to Panel p

        this.add(cboxPanel,"West");

        // Set the current checkbox states

        for(int i=1; i<=8; i++) {
            checkBox[i].setState(
                DataHolder.includeReaction[IsotopePad.protonNumber][IsotopePad.neutronNumber][i]);
        }

        // Add scrollpane window with detailed reaction list

        ScrollPane sp = new ScrollPane(ScrollPane.SCROLLBARS_AS_NEEDED);
        sp.setBackground(panelBackColor2);
        sp.setForeground(panelForeColor);
        rl = new ReactionList();
        Panel rlist = new Panel();

        // Set a GridBagLayout for the Panel rlist and define a
        // GridBagConstraints instance that will hold the constraints
        // for components in this layout.

        rlist.setLayout(new GridBagLayout());
        GridBagConstraints constraints = new GridBagConstraints();

        // Set the relevant fields of the constraint object
        constraints.weightx = constraints.weighty = 100.0;  // Relative weights
        constraints.gridx = 0;                                            // x position for component
        constraints.gridy = 0;                                     // y position for component
        constraints.ipadx = 0;                                    // Internal padding for component in x
        constraints.ipady = 0;                                    // Internal padding for component in y
        constraints.anchor = GridBagConstraints.NORTHWEST; // Anchor upper left
        constraints.insets = new Insets(5,10,5,10);  // Margins
        rlist.add(rl,constraints);                         // Add ReactionList rl to Panel rlist,
                                                                  // constrained by constraint object.
        sp.add(rlist);                                        // Add Panel rlist to ScrollPane sp
        this.add("Center",sp);                          // Add ScrollPane sp to the current object

        // Add Reset, Cancel, Save Changes, and Print buttons

        Panel botPanel = new Panel();
        botPanel.setFont(buttonFont);
        botPanel.setBackground(MyColors.gray204);

        Button resetButton = new Button(" Reset ");
        Button dismissButton = new Button("Cancel");
        Button saveButton = new Button("Save Changes");
        Button printButton = new Button("  Print  ");
        botPanel.add(resetButton);
        botPanel.add(dismissButton);
        botPanel.add(saveButton);
        botPanel.add(printButton);

        this.add("South", botPanel);

        // Add inner class event handler for Dismiss button

        dismissButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
              hide();
              dispose();
            }
        });


        // Add inner class event handler for Reset button

        resetButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){

                // Set the reaction class flags according to the
                // current reaction class checkbox states

                for(int i=1; i<=8; i++) {
                    DataHolder.includeReaction
                       [IsotopePad.protonNumber]
                       [IsotopePad.neutronNumber][i]
                           = checkBox[i].getState();
                }

                upDateComponents();
            }
        });


        // Add inner class event handler for Save Changes button

        saveButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){

                // Set the reaction class flags according to the
                // current reaction class checkbox states

                for(int i=1; i<=8; i++) {
                    DataHolder.includeReaction[IsotopePad.protonNumber][IsotopePad.neutronNumber][i] 
                        = checkBox[i].getState();
                }

                // Ensure that any changes in checkboxes are recorded
                // before states are used to set DataHolder.RnotActive
                // array.

                upDateComponents();

                // Store state of updated reaction component checkboxes.  If
                // showing and checked, set DataHolder.RnotActive[Z][N][i]
                // to false (which means allow the reaction to contribute),
                // but if showing and not checked, or not showing (meaning
                // that the reaction category is de-selected for this isotope)
                // set to true, which suppresses the reaction.

                for (int i=0; i<lenny; i++) {

                    if( rl.cb[i].isShowing() ) {
                        DataHolder.RnotActive[Z][N][i] = !rl.cb[i].getState();
                    } else {
                        DataHolder.RnotActive[Z][N][i] = true;
                    }
                    System.out.println("i="+i+" Z="+Z+" N="+N
                        +" "+DataHolder.RnotActive[Z][N][i]+"  "+rl.rArray[i].reacString);
                }

                DataHolder.wasOpened[Z][N] = true;

                hide();
                dispose();
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
              hide();
              dispose();
           }
        });
    }



    // ---------------------------------------------------------------------------------
    //  Method upDateComponents to store state of updated
    //  reaction component checkboxes
    // ---------------------------------------------------------------------------------

    void upDateComponents() {

        //  The length of the list is the total reactions rl.len for heavies, but for light-light only the
        //  subset of all reactions that are light-light (not already included under a heavy isotope)
        //  is incuded in the list.
        
        lenny = rl.len;
        if(Z==1 && N==0) lenny = 17;
        if(Z==2 && N==2) lenny = 41;
        if(Z==0 && N==1) lenny = 19;
        if(Z==1 && N==1) lenny = 24;
        if(Z==1 && N==2) lenny = 19;
        if(Z==2 && N==1) lenny = 16;
        
        for (int i=0; i<lenny; i++) {
            if( rl.cb[i].isShowing()) {
                DataHolder.RnotActive[Z][N][i] = !rl.cb[i].getState();
            }
        }

        // Add or remove individual reactions from the list
        // if the corresponding reaction classes have changed

        for (int i=0; i<lenny; i++) {
            int rc = rl.rArray[i].reacIndex;    // Reaction class
            if(checkBox[rc].getState()) {     // If class selected
                rl.add(rl.cb[i]);
                rl.cb[i].setState(!DataHolder.RnotActive[Z][N][i]);
            } else {                                   // If not selected
                rl.remove(rl.cb[i]);
            }
        }

        show();        //Force redisplay with changed components

    }



    // ---------------------------------------------------------------------------------------------
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
    // -----------------------------------------------------------------------------------------------

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

}   /* End class IsoData */

