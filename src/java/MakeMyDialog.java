// ----------------------------------------------------------------------------------------------------------
//  Example class for testing creation of a modally blocking dialog window.
//  Test by executing java MakeMyDialog.
// ----------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;

public class MakeMyDialog extends Frame {

    Font warnFont = new java.awt.Font("SanSerif", Font.BOLD, 12);
    FontMetrics warnFontMetrics = getFontMetrics(warnFont);


    public static void main (String[] args) {

        MakeMyDialog md = new MakeMyDialog(50,50,300,100,Color.black,
            Color.white,"A Frame", "A frame from which warning is launched",true);
    }


    // ------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------
    
    public MakeMyDialog (int X, int Y, int width, int height, Color fg, Color bg, String title,
        String text, boolean oneLine) {

        // Create a frame from which we will launch a dialog warning window
    
        final Frame mw = new Frame(title);
        mw.setLayout(new BorderLayout());
        mw.setSize(width,height);
        mw.setLocation(X,Y);
        Label hT = new Label(text,Label.CENTER);
        hT.setForeground(fg);
        hT.setBackground(bg);
        hT.setFont(warnFont);
        mw.add("Center", hT);
        mw.setTitle(title);
    
        // Add show warning button
    
        Panel botPanel = new Panel();
        botPanel.setBackground(Color.lightGray);
        Label label1 = new Label();
        Label label2 = new Label();
    
        Button showButton = new Button("Show Modal Warning Window");
        botPanel.add(label1);
        botPanel.add(showButton);
        botPanel.add(label2);
    
        // Add inner class event handler for show warning button.  This must be
        // added to the showButton before botPanel is added to mw.
    
        showButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){ 
                MakeMyDialog.this.makeTheWarning(200,200,200,100,Color.black,
                    Color.lightGray,"Warning","Warning message",
                    true, MakeMyDialog.this);
            }
        });
    
        mw.add("South", botPanel);
    
        // Add window closing button (inner class)
    
        mw.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                mw.hide();
                mw.dispose();
                System.exit(0);
            }
        });
    
        mw.show();
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
    // -------------------------------------------------------------------------------------------------------------------------

    public void makeTheWarning (int X, int Y, int width, int height,
        Color fg, Color bg, String title,
        String text, boolean oneLine, Frame frame) {

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

}  /* End class MakeMyDialog */
