// --------------------------------------------------------------------
//  Class to create Help Window
// --------------------------------------------------------------------


import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;


class MyHelpFrame extends Frame {

    // ----------------------------------------------------------------
    //  Constructor
    // ----------------------------------------------------------------
    
    public MyHelpFrame() {
    
        Font textFont = new java.awt.Font("SanSerif", Font.PLAIN, 12);
        FontMetrics textFontMetrics = getFontMetrics(textFont);
    
        setLayout(new BorderLayout());
    
        TextArea hT = new TextArea("",300,50,TextArea.SCROLLBARS_VERTICAL_ONLY);
        hT.setEditable(false);
    
        // Set color and font for TextArea.  Note that a setBackground(Color)
        // seems to have no effect on some machines once setEditable(false)
        // is set above.
    
        hT.setForeground(Color.black);
        hT.setFont(textFont);
    
        this.add("Center", hT);
    
        String helpString = "\nThis tool allows one to choose graphically the";
        helpString += " isotopes and the classes of reaction involving those";
        helpString += " isotopes in a reaction network calculation, to set relevant parameters,";
        helpString += " run the calculation, and then visualize the results graphically.";
        helpString += "\n\nReaction Class\nThere are 8 classes";
        helpString += " of reactions.  The classes to include in the";
        helpString += " calculation are selected using the checkboxes on";
        helpString += " the right side (the checkboxes may all be cleared";
        helpString += " with the Clear button).\n\n";
        helpString += "Box Size\nControls the size of isotope boxes in the display.  Click";
        helpString += " \"Reset\" after changing to redisplay.\n\n";
        helpString += "Max Z and Max N\nControl the max proton and neutron numbers to be";
        helpString += " displayed (click \"Reset\" after changing).  Max Z and Max N generally cannot";
        helpString += " be larger than the variables pmax and pmin, respectively, defined";
        helpString += " in the class StochasticElements.";
        helpString += "\n\nShow Labels\nToggles isotope labels if box sizes are medium or large.";
        helpString += "\n\nSelect Isotopes\nOpens windows to allow selection of active istotopes";
        helpString += " and rates for calculation. You must select reaction classes before clicking";
        helpString += " Select Isotopes.";
        helpString += "\n\nCalculate\nInitiates calculation after active istotopes, initial abundances,";
        helpString += " and parameters are set.";
        helpString += "\n\nSet Parameters\nOpens window to set parameters.";
        helpString += "\n\nPlot Rates\nOpens window permitting reaction rates to be plotted for";
        helpString += " isotopes selected. Select isotopes first by left clicking on corresponding box.";
        helpString += " Don't try for light ions like H or He (too many reactions; if you want to plot rates";
        helpString += " for say He4 + C12 -> ?, select C12, not He4) and don't select";
        helpString += " more than a few isotopes at a time (again, too many reactions).\n\n";
    
        hT.appendText(helpString);
    
        // Add window closing button (inner class)
    
        this.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                hide();
                dispose();
                SegreFrame.helpWindowOpen = false;
            }
        });
    }

}  /* End class MyHelpFrame */

