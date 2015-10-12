// --------------------------------------------------------------------------------------------------------
//  Class ContourPlotter to generate contour plot of isotopic abundances.
//  Uses ContourDisplay and GraphicsPad classes.
// --------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Vector;
import java.util.Properties;


class ContourPlotter extends Frame {
    private ContourDisplay contour;
    private GraphicsPad gp;

    public ContourPlotter() {
        setLayout(new BorderLayout());
        contour = new ContourDisplay();
        gp = new GraphicsPad();
        add("Center", gp);
        add("South", contour);

        this.addWindowListener(new WindowAdapter() {
          public void windowClosing(WindowEvent e) {
            hide();
            dispose();
          }
        });
    }

}   /* End class ContourPlotter */

