package edu.utk.phys.fern;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Insets;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextArea;

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