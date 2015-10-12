// ---------------------------------------------------------------------------------------
// Class GenericHelpFrame to create generic Help Window
//     s = Text string describing the help (\n to skip lines)
//     t = Text for title bar
//     h = height of help window
//     w = width of help window
//     X = screen x position of help window
//     Y = sceen y position of help window
//     lines = lines argument for embedded TextArea
//     columns = column argument for embedded TextArea
// ---------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

class GenericHelpFrame extends Frame {

   // ---------------------------------------------------------------
   //  Public constructor
   // ---------------------------------------------------------------

   public GenericHelpFrame(String s, String t, int h, int w, int X,int Y, int lines, int columns) {

      Font textFont = new java.awt.Font("SanSerif", Font.PLAIN, 12);
      FontMetrics textFontMetrics = getFontMetrics(textFont);
      Font buttonFont = new java.awt.Font("SanSerif", Font.BOLD, 11);
      FontMetrics buttonFontMetrics = getFontMetrics(buttonFont);
      Color bgColor = MyColors.helpColor; //new Color(245,245,245);

      this.pack();
      this.setSize(w,h);
      this.setLocation(X,Y);
      this.setTitle(t);
      this.setResizable(false);

      setLayout(new BorderLayout());

      TextArea hT = new TextArea("",lines,columns, TextArea.SCROLLBARS_NONE);
      hT.setEditable(false);

      // Set color and font for TextArea.

      hT.setForeground(Color.black);
      hT.setFont(textFont);
      hT.setBackground(bgColor);
      // hT.disable();

      ScrollPane sp = new ScrollPane(ScrollPane.SCROLLBARS_AS_NEEDED);
      sp.add(hT);
      this.add("Center", sp);

      Panel leftPanel = new Panel();
      leftPanel.setBackground(MyColors.gray204);

      this.add("West",leftPanel);

      // Add Dismiss button and bottom panel

      Panel botPanel = new Panel();
      botPanel.setFont(buttonFont);
      botPanel.setBackground(MyColors.gray204);
      Button dismissButton = new Button("Cancel");
      botPanel.add(dismissButton);
      this.add("South", botPanel);

      // Insert the text string in the TextArea

      hT.appendText("\n" + s);

      // Add inner class event handler for Dismiss button

      dismissButton.addActionListener(new ActionListener() {
          public void actionPerformed(ActionEvent ae){
              hide();
              dispose();
              ParamSetup.helpWindowOpen = false;
          }
      });

      // Add window closing button (inner class)

      this.addWindowListener(new WindowAdapter() {
         public void windowClosing(WindowEvent e) {
            hide();
            dispose();
            ParamSetup.helpWindowOpen = false;
         }
      });
   }

}  /* End class GenericHelpFrame */

