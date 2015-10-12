package edu.utk.phys.fern;
// --------------------------------------------------------------------------------------------------------
//  Class ContourPlotter to generate contour plot of isotopic abundances.
//  Uses ContourDisplay and GraphicsPad classes.
// --------------------------------------------------------------------------------------------------------

import java.awt.BorderLayout;
import java.awt.Frame;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;


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

