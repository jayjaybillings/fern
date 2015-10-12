// ------------------------------------------------------------------------------------------
//  Class to create generic file dialog window.  Variables:
//     X = x-position of window (referenced to upper left)
//     Y = y-position of window (referenced to upper left)
//     width = width of window
//     height = height of window
//     fg = foreground (font) color
//     bg = background color
//     title = title for window
//     text = text for window
//  The filename specified by user is given by hT.getText().
// -------------------------------------------------------------------------------------------


import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

class PlotFileDialogue extends Frame {

    Font warnFont = new java.awt.Font("SanSerif", Font.BOLD, 12);
    FontMetrics warnFontMetrics = getFontMetrics(warnFont);
    Font buttonFont = new java.awt.Font("SanSerif", Font.BOLD, 11);
    FontMetrics buttonFontMetrics = getFontMetrics(buttonFont);
    Font inputFont = new java.awt.Font("Monospaced", Font.PLAIN, 12);
    FontMetrics inputFontMetrics = getFontMetrics(inputFont);
    
    TextField hT = new TextField();
    
    public PlotFileDialogue(int X, int Y, int width, int height, Color fg,
        Color bg, String title, String text) {

        this.setLayout(new BorderLayout());
        this.setSize(width,height);
        this.setLocation(X,Y);
        
        Label topLabel = new Label(text,Label.CENTER);
        topLabel.setFont(warnFont);
        topLabel.setForeground(fg);
        topLabel.setBackground(bg);
        
        Panel midPanel = new Panel();
        midPanel.setBackground(bg);
        midPanel.setLayout(null);
        midPanel.add(hT);
        hT.setSize(width,25);
        this.setTitle(title);
        
        hT.setForeground(fg);
        hT.setBackground(Color.white);
        hT.setFont(inputFont);
        this.add("North",topLabel);
        this.add("Center", midPanel);
        
        // Add Cancel and Save buttons
        
        Panel botPanel = new Panel();
        botPanel.setBackground(bg);
        botPanel.setFont(buttonFont);
        Label label1 = new Label();
        Label label2 = new Label();
        Label label3 = new Label();
        label2.setBackground(bg);
        label3.setBackground(bg);
        
        Button dismissButton = new Button("Cancel");
        Button saveButton = new Button("Save");
        botPanel.add(dismissButton);
        botPanel.add(label1);
        botPanel.add(saveButton);
        
        this.add("South", botPanel);
        this.add("West",label2);
        this.add("East",label3);
        

        // Add inner class event handler for Dismiss button
        
        dismissButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
                hide();
                dispose();
            }
        });  // -- end inner class for Dismiss button processing


        // Add inner class event handler for postscript Save button
        
        saveButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
        
                // Following invokes gp.PSfile() to print the PS
                // file.  Can't invoke this method directly from
                // here because object gp is not recognized.  This
                // works by declaring AbPlotFrame instance cd and
                // AbGraphicsPad instance gp to be static and then
                // invoking the static method makePs from AbPlotFrame
                // to in turn invoke gp.PSfile(file).  (Note the use
                // of the "PlotFileDialogue.this" construction to
                // invoke the "this" reference for the outer class
                // PlotFileDialogue from this anonymous inner class.)
                // In other applications, the second line of the if
                // clause would likely require modification to
                // correspond to the particular classes in use.
        
                if ( PlotFileDialogue.this.hT.getText().length() > 0 ) {
                    AbPlotFrame.makePS(PlotFileDialogue.this.hT.getText());
                    hide();
                    dispose();
                } else {
        
                    makeTheWarning(100,100,250,100,Color.black,
                        Color.lightGray, "Warning",
                        "You must supply a file name!",
                        true, PlotFileDialogue.this);
        
                    return;
                }
            }
        });  // -- end inner class for PS Save button processing


        // Add window button (inner class)
        
        this.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                hide();
                dispose();
            }
        });
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
            TextArea hT = new TextArea("",height,width,
                TextArea.SCROLLBARS_NONE);
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


}  /* End class PlotFileDialogue */
