
//  The class MakeWarning creates a modal warning window when invoked
//  from within an object that subclasses Frame. The window is
//  modally blocked (the window from which the warning window is
//  launched is blocked from further input until the warning window
//  is dismissed by the user).  Method arguments:
//
//      X = x-position of window (relative to container upper left)
//      Y = y-position of window (relative to container upper left)
//      width = width of window
//      height = height of window
//      title = title string
//      text = warning string text
//      frame = A Frame or JFrame that is the parent window modally
//              blocked by the warning window.  If the parent
//              class from which this method is invoked extends
//              Frame, this argument can be just "this" (or
//              "ParentClass.this" if invoked from an
//              inner class event handler of ParentClass).
//              Otherwise, it must be the name of an object derived from
//              Frame that represents the window modally blocked.
//
//  Invoke from another class using the constructor.  For example,
//
//      new MakeWarning (100, 100, 180, 140, "ERROR!", 
//              "You really did it this time.", this);
// 
//  But remember:  if invoked from an inner class (e.g., an event handler
//  defined in an anonymous inner class), the last argument should be 
//  ParentClass.this, where ParentClass is the class enclosing the inner class
//  invoking the warning window.
    

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import javax.swing.*;

class MakeWarning {

    public MakeWarning (int X, int Y, int width, int height,
		    String title, String text, JFrame frame) {

	// Find position on the screen of the container launching the
	// modal window so that we can position the warning dialog window
        // relative to that location.
	
	Point containerLocation = frame.getLocationOnScreen();

	// Create Dialog window with modal blocking set to true.
	// Make final so inner class below can access it.
	
	final JDialog mww = new JDialog(frame, title, true);
	mww.pack();
	mww.setLayout(new BorderLayout());
	mww.setSize(width,height);
	// Offset relative to container upper left
	mww.setLocation(containerLocation.x + X, containerLocation.y + Y);

	JTextArea hT = new JTextArea(text);
		    
	hT.setEditable(false);
	hT.setLineWrap(true);
	hT.setWrapStyleWord(true);
	hT.setMargin( new Insets(10,10,0,10) ); // Order: top, left, bottom, right		
	mww.add(hT, BorderLayout.CENTER);

	mww.setTitle(title);

	// Add dismiss button

	JPanel botPanel = new JPanel();
	botPanel.setBackground(Color.lightGray);
	JLabel label1 = new JLabel();
	JLabel label2 = new JLabel();

	JButton dismissButton = new JButton("Dismiss");
	botPanel.add(label1);
	botPanel.add(dismissButton);
	botPanel.add(label2);

	mww.add("South", botPanel);
	
	// Add inner class event handler for Dismiss button.  

	dismissButton.addActionListener(new ActionListener() {
	    public void actionPerformed(ActionEvent ae){
	      mww.hide();
	      mww.dispose();
	    }
	});

	// Add window closing button

	mww.addWindowListener(new WindowAdapter() {
	  public void windowClosing(WindowEvent e) {
	      mww.hide();
	      mww.dispose();
	  }
	});
	
	// Note that following show () must come after all the above
	// additions; otherwise they are not added before the
	// window is displayed.

	mww.show();

    }

}