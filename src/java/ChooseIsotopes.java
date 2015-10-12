// ---------------------------------------------------------------------------------------------------------------
//  Class ChooseIsotopes to lay out main controlling frame for ElementMaker
// ---------------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

public class ChooseIsotopes {

    static Color segreBC = Color.black; // Background color

    // SegreFrame instance declared static in order to allow
    // instances of MyFileDialogue class to communicate with
    // the instance gp of IsotopePad through a SegreFrame
    // method (PSfile).  See the MyFileDialogue class, and the later
    // static declaration of IsotopePad gp = new IsotopePad()

    static SegreFrame cd = new SegreFrame();

    // ----------------------------------------------------------------
    //  Public constructor
    // ----------------------------------------------------------------

    public ChooseIsotopes() {

        // Create a customized main frame and display it

        cd.setSize(800,600);
        cd.setTitle(" Isotope Selection");
        cd.setResizable(false);
        cd.setBackground(segreBC);
        cd.setLocation(50,50);

        //  Create a menu bar and add menu to it

        MenuBar cdmb = new MenuBar();
        cd.setMenuBar(cdmb);
        Menu cdMenu = new Menu("File");
        cdmb.add(cdMenu);
        Menu helpMenu = new Menu("Help");
        cdmb.add(helpMenu);

        // Create menu items with keyboard shortcuts

        MenuItem s,l,p,h,q;
        cdMenu.add(s=new MenuItem("Save as Postscript",new MenuShortcut(KeyEvent.VK_S)));
        cdMenu.add(l=new MenuItem("Load", new MenuShortcut(KeyEvent.VK_L)));
        cdMenu.add(p=new MenuItem("Print", new MenuShortcut(KeyEvent.VK_P)));
        cdMenu.addSeparator();     //  Menu separator
        cdMenu.add(q=new MenuItem("Quit", new MenuShortcut(KeyEvent.VK_Q)));

        helpMenu.add(h=new MenuItem("Instructions", new MenuShortcut(KeyEvent.VK_H)));

        // Create and register action listeners for menu items

        s.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e){
              MyFileDialogue fd =
                  new MyFileDialogue(100,100,400,110,Color.black,
                          MyColors.gray204,"Choose File Name",
                          "Choose a postscript file name:");
              fd.setResizable(false);
              fd.show();
            }
        });

        l.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e){System.exit(0);}
        });

        p.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e){
                cd.printThisFrame(20,20,false);}
        });

        h.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e){
              if(cd.helpWindowOpen) {
                  cd.hf.toFront();
              } else {
                  cd.hf = new MyHelpFrame();
                  cd.hf.setSize(300,400);
                  cd.hf.setLocation(100,100);
                  cd.hf.setResizable(false);
                  cd.hf.setTitle(" Help");
                  cd.hf.show();
                  cd.helpWindowOpen = true;
              }
            }
        });

        q.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e){System.exit(0);}
        });

          cd.show();
	}

}  /* End class ChooseIsotopes */



