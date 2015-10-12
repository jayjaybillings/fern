package edu.utk.phys.fern;
// ---------------------------------------------------------------------------
//  Class ContourDisplay used by ContourPlotter
// ---------------------------------------------------------------------------

import java.awt.Canvas;
import java.awt.Panel;
import java.awt.TextField;


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

