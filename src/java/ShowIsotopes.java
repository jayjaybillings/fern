// -------------------------------------------------------------------------------------------------------------
//  Class ShowIsotopes to lay out contour plotting frame for ElementMaker
//  by creating an instance of ContourFrame
// -------------------------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

public class ShowIsotopes {

    static Color isoBC = Color.black;     //MyColors.gray235; // Background color

    // ContourFrame instance declared static in order to allow
    // instances of MyFileDialogue class to communicate with
    // the instance gp of ContourPad through a ContourFrame
    // method (PSfile).  See the MyFileDialogue class, and the
    // static declaration of ContourPad gp = new ContourPad()

    static ContourFrame cd;


    // ----------------------------------------------------------------
    //  Public constructor
    // ----------------------------------------------------------------

    public ShowIsotopes() {

         cd = new ContourFrame();

        // Create a customized main frame and display it

        cd.setSize(800,600);
        cd.setTitle(" Isotopic Abundances");
        cd.setBackground(isoBC);
        cd.setLocation(200,200);

        //  Create a menu bar and add menu to it

        MenuBar cdmb = new MenuBar();
        cd.setMenuBar(cdmb);
        Menu cdMenu = new Menu("File");
        cdmb.add(cdMenu);
        Menu helpMenu = new Menu("Help");
        cdmb.add(helpMenu);

        // Create menu items with keyboard shortcuts

        MenuItem s,m,l,p,h,q;
        cdMenu.add(s=new MenuItem("Save as Postscript",
            new MenuShortcut(KeyEvent.VK_S)));
        cdMenu.add(m=new MenuItem("Save Movie Frames",
            new MenuShortcut(KeyEvent.VK_M)));
        cdMenu.add(l=new MenuItem("Load", new MenuShortcut(KeyEvent.VK_L)));
        cdMenu.add(p=new MenuItem("Print", new MenuShortcut(KeyEvent.VK_P)));
        cdMenu.addSeparator();     //  Menu separator
        cdMenu.add(q=new MenuItem("Quit", new MenuShortcut(KeyEvent.VK_Q)));

        helpMenu.add(h=new MenuItem("Instructions",
            new MenuShortcut(KeyEvent.VK_H)));

        // Create and register action listeners for menu items

        s.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e){
              ContourFileDialogue fd =
                  new ContourFileDialogue(100,130,400,110,Color.black,
                    MyColors.gray204,"Choose File Name",
                    "Choose a postscript file name:");
              fd.setResizable(false);
              fd.hT.setText("nz.ps");
              fd.show();
            }
        });

        m.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e){
                cd.gp.saveFrames();}
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

}  /* End class ShowIsotopes */



