// --------------------------------------------------------------------------------------------------------------
//  Class ShowAbundances to pop up window to display current abundances
//  Y and mass fractions X for isotopes.
// --------------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import javax.swing.*;


class ShowAbundances extends JFrame  {

    Font titleFont = new java.awt.Font("SanSerif", Font.BOLD, 12);
    FontMetrics titleFontMetrics = getFontMetrics(titleFont);
    Font buttonFont = new java.awt.Font("SanSerif", Font.BOLD, 10);
    FontMetrics buttonFontMetrics = getFontMetrics(buttonFont);
    Font textFont = new java.awt.Font("SanSerif", Font.PLAIN, 12);
    FontMetrics textFontMetrics = getFontMetrics(textFont);

    Color panelForeColor = Color.black;
    Color panelBackColor = new Color(230,230,230);
    Color disablebgColor = new Color(220,220,220);
    Color disablefgColor = new Color(180,180,180);
    Color textfieldColor = new Color(255,255,255);

    JPanel panel0, panel1;
    JPanel cboxPanel;
    JTextField massFrac,Y;
    JLabel massFracL, YL;

    int ZZ, NN;


    // ---------------------------------------------------------------------
    //  Public constructor
    // ---------------------------------------------------------------------

    public ShowAbundances (int width, int height, String title, String text){
    
		// Bug fix, see http://bugs.sun.com/bugdatabase/view_bug.do?bug_id=7027598
		// and http://jasperforge.org/plugins/espforum/view.php?group_id=102&forumid=103&topicid=83582
		massFrac.setDropTarget(null);
		Y.setDropTarget(null);
		
        ZZ = IsotopePad.protonNumber;
        NN = IsotopePad.neutronNumber;
        String symbol = Cvert.returnSymbol(ZZ);
        String arg = "("+ZZ+","+NN+")";
        
        //String nn = "Z="+String.valueOf(ZZ);
        //nn += "  N="+String.valueOf(NN);
        String nn = (" " +(ZZ+NN)+"-"+symbol);
        nn+= " (t="+Cvert.gg.decimalPlace(5,ContourPad.currentTime)+" s)";

        this.pack();
        this.setSize(width,height);
        this.setTitle(title);

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
        gs.insets = new Insets(10,10,10,10);

        // Main container (all but bottom button panel)

        cboxPanel = new JPanel();
        cboxPanel.setLayout(new GridBagLayout());

        // Top label

        JLabel reacLabel = new JLabel(nn,JLabel.CENTER);
        reacLabel.setFont(titleFont);
        gs.gridwidth = 1;
        cboxPanel.add(reacLabel,gs);

        // Mass fraction widgets on panel0

        gs.anchor = GridBagConstraints.CENTER;
        gs.gridy = GridBagConstraints.RELATIVE;

        panel0 = new JPanel();

        massFracL = new JLabel("Mass fraction X"+arg, JLabel.LEFT);
        //massFracL = new JLabel("Mass Fraction X(Z,N): ",JLabel.LEFT);

        massFracL.setFont(textFont);
        panel0.add(massFracL);

        massFrac = new JTextField(10);
        massFrac.setFont(textFont);
        massFrac.setEditable(false);
        panel0.add(massFrac);

        cboxPanel.add(panel0,gs);

        // Abundance Y(i) widgets on panel1

        gs.gridy =GridBagConstraints.RELATIVE ;

        panel1 = new JPanel();

        YL = new JLabel("Abundance Y"+arg, JLabel.LEFT);
        YL.setFont(textFont);
        panel1.add(YL);

        Y = new JTextField(10);
        Y.setFont(textFont);
        Y.setEditable(false);
        panel1.add(Y);

        cboxPanel.add(panel1, gs);

        // Add the container for mass fractions and abundances to the frame

        this.add(cboxPanel,"Center");

        // Put stuff on the bottom panel botPanel and add to frame

        JPanel botPanel = new JPanel();
        botPanel.setBackground(MyColors.gray204);

        JButton dismissButton = new JButton("Cancel");
        dismissButton.setFont(buttonFont);
        botPanel.add(dismissButton);

        this.add("South", botPanel);
        
        // Read any values already set and place in fields

        String temp, temp2;
        if(StochasticElements.Y[ZZ][NN] != 0) {
            temp = Cvert.gg.decimalPlace(7,(StochasticElements.Y[ZZ][NN]));
            temp2 = Cvert.gg.decimalPlace(7,(StochasticElements.Y[ZZ][NN]*(double)(ZZ + NN)));
        } else {temp2 = temp = "";}
        Y.setText(temp);
        massFrac.setText(temp2);

        // Add inner class event handler for Dismiss button

        dismissButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
            hide();
            dispose();
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
    

}   /* End class ShowAbundances */

