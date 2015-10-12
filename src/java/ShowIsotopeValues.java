// --------------------------------------------------------------------------------------------------------------
//  Class ShowIsotopeValues to pop up window to display current value of
//  the variable being displayed
// --------------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import javax.swing.*;


class ShowIsotopeValues extends JFrame  {

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
    JTextField variable,Y;
    JLabel variableL, YL;

    int ZZ, NN;


    // ---------------------------------------------------------------------
    //  Public constructor
    // ---------------------------------------------------------------------

    public ShowIsotopeValues (int width, int height, String title, String text, int t, double [][][] twa){
    
		// Bug fix, see http://bugs.sun.com/bugdatabase/view_bug.do?bug_id=7027598
		// and http://jasperforge.org/plugins/espforum/view.php?group_id=102&forumid=103&topicid=83582
		variable.setDropTarget(null);
		Y.setDropTarget(null);
		
        ZZ = IsotopePad.protonNumber;
        NN = IsotopePad.neutronNumber;
        String symbol = Cvert.returnSymbol(ZZ);
        //String arg = "("+ZZ+","+NN+")";
        String nn = (" " +(ZZ+NN)+"-"+symbol);
        nn+= " (t="+Cvert.gg.decimalPlace(5,StochasticElements.timeNow[t])+" s)";

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

        variableL = new JLabel(text, JLabel.LEFT);
        variableL.setFont(textFont);
        panel0.add(variableL);

        variable = new JTextField(9);
        variable.setFont(textFont);
        variable.setEditable(false);
        panel0.add(variable);

        cboxPanel.add(panel0,gs);

        // Add the container for the value of the variable to the frame

        this.add(cboxPanel,"Center");

        // Put stuff on the bottom panel botPanel and add to frame

        JPanel botPanel = new JPanel();
        botPanel.setBackground(MyColors.gray204);

        JButton dismissButton = new JButton("Cancel");
        dismissButton.setFont(buttonFont);
        botPanel.add(dismissButton);

        this.add("South", botPanel);
        
        // Read any values already set and place in fields

        String temp;
        if(twa[ZZ][NN][t] != 0) {
            temp = Cvert.gg.decimalPlace(7, twa[ZZ][NN][t]);
        } else {temp = "";}
        variable.setText(temp);

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
    

}   /* End class ShowIsotopeValues */

