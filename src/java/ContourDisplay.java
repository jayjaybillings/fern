// ---------------------------------------------------------------------------
//  Class ContourDisplay used by ContourPlotter
// ---------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Vector;
import java.util.Properties;


class ContourDisplay extends Panel {
    private TextField show;

    public ContourDisplay() {
        show = new TextField(10);
        Canvas contourCanvas;
        Panel p = new Panel();
        p.add(show);
        add("Center", p);
    }
}

