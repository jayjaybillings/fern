
// Class GenericHelpDialog to create generic Help Window.
//
//     s = Text string describing the help (\n to skip lines)
//     t = Text for title bar
//     h = height of help window
//     w = width of help window
//     X = screen x position of help window
//     Y = sceen y position of help window
//     frame = component (which must subclass Frame oe JFrame) launching 
//     help dialog
//
//  Note that if this dialog is invoked from an inner class of the parent
//  Frame class ParentClass (for example, from an anonymous inner class 
//  event handler), then the frame argument should be ParentClass.this
//  instead of this.


import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import javax.swing.*;


class GenericHelpDialog extends JFrame {

   // ---------------------------------------------------------------
   //  Public constructor
   // ---------------------------------------------------------------

   public GenericHelpDialog(String s, String t, int h, int w, int X, int Y, Frame frame) {
   
      // Find position on the screen of the container launching the
      // modal window so that we can position the help dialog window
      // relative to that location.
	
      Point containerLocation = frame.getLocationOnScreen();

      this.pack();
      this.setSize(w,h);
      this.setLocation(containerLocation.x + X, containerLocation.y + Y);
      this.setTitle(t);
      this.setResizable(false);

      setLayout(new BorderLayout());

      JTextArea hT = new JTextArea(s);
      hT.setEditable(false);
      hT.setLineWrap(true);
      hT.setWrapStyleWord(true);
      hT.setMargin( new Insets(10,10,0,10) ); // Order: top, left, bottom, right

      JScrollPane sp = new JScrollPane(hT, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED, 
		 JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);

      this.add(sp, BorderLayout.CENTER);

      JPanel leftPanel = new JPanel();
      leftPanel.setBackground(Color.lightGray);

      this.add("West",leftPanel);

      // Add Dismiss button and bottom panel

      JPanel botPanel = new JPanel();
      botPanel.setBackground(Color.lightGray);
      JButton dismissButton = new JButton("Cancel");
      botPanel.add(dismissButton);
      this.add("South", botPanel);

      
      // Add inner class event handler for Dismiss button

      dismissButton.addActionListener(new ActionListener() {
          public void actionPerformed(ActionEvent ae){
              hide();
              dispose();
              // Following line specific to this application
              UI.helpWindowOpen = false;
          }
      });

      // Add window closing button

      this.addWindowListener(new WindowAdapter() {
         public void windowClosing(WindowEvent e) {
            hide();
            dispose();
            // Following line specific to this application
            UI.helpWindowOpen = false;
         }
      });
   }

}

