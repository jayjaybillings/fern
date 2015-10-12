// -------------------------------------------------------------------------------------------------------------------------
//  Class ProgressMeter creates a progress meter window. Arguments:
//
//      X = x position on screen of dialog window
//      Y = y position on screen of dialog window
//      width = width of window
//      height = height of window
//      tbString = String for title bar
//      s1 = top string displayed in window
//      s2 = bottom string displayed in window
//
//  Usage:  create an instance
//
//      ProgressMeter pm = new ProgressMeter(X,Y,w,h,s0,s1,s2);
//
//  This creates the window with geometry and initial strings determined
//  by the arguments.  You may then modify the strings s1 and s2 to indicate
//  progress by commands of the form
//
//      pm.sets1(String news1);
//      pm.sets2(String news2);
//
//  (It is up to the user to construct the appropriate strings and to
//   display at the appropriate times.)  When the window is no longer
//  needed, close it and release its resources by
//
//      pm.quit();
//
// -------------------------------------------------------------------------------------------------------------------------


import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;

class ProgressMeter extends Frame implements Runnable {

    Font buttonFont = new java.awt.Font("Arial", Font.BOLD, 11);
    FontMetrics buttonFontMetrics = getFontMetrics(buttonFont);

    final Dialog dg;
    ProgressPad cs;
    Graphics g;
    Label text;
    Label time;
    Thread runner = null;
    boolean runThread = true;
    int sleepTime = 100;
    String s1;
    String s2;


    // --------------------------------------
    //  Public constructor
    // --------------------------------------

    public ProgressMeter(int X, int Y, int width, int height, String tbString, String s1, String s2) {

        this.s1 = s1;
        this.s2 = s2;

        dg = new Dialog(this);
        dg.setSize(width,height);
        dg.setLocation(X,Y);
        dg.setBackground(new Color(204,204,204));
        dg.setFont(buttonFont);
        dg.setTitle(tbString);

        text = new Label(s1, Label.CENTER);
        cs = new ProgressPad();
        cs.setBackground(Color.red);
        time = new Label(s2, Label.CENTER);

        dg.add(text, "North");
        dg.add(time, "Center");

        Panel botPanel = new Panel();
        Button cancelButton = new Button("Cancel");
        Button stopButton = new Button ("Stop");
        botPanel.add(cancelButton);
        botPanel.add(stopButton);
        dg.add(botPanel, "South");

        dg.show();

        startThread();

        // Button action to cancel this window.  Handle with an inner class.

        cancelButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){

                System.out.println("Cancel");
                runThread = false;
                dg.hide();
                dg.dispose();
            }
        });

    }

    public void run() {

        while(runThread) {
            //cs.repaint();
            try{Thread.sleep(sleepTime);} catch (InterruptedException e){;}
        }

    }


    // ----------------------------------------------------------------
    //  Method to start the animation thread
    // ----------------------------------------------------------------

    public void startThread() {
        runner = new Thread(this);
        runner.setPriority(Thread.NORM_PRIORITY);
        runner.start();
    }


    // ------------------------------------------------------------------------------------------------
    //  Following public methods allow the text label strings to be
    //  modified to indicate progress in the process being monitored
    // ------------------------------------------------------------------------------------------------

    public void sets1(String s) {
        text.setText(s);
    }

    public void sets2(String s) {
        time.setText(s);
    }


    // -------------------------------------------------------------------------------------------------------
    //  This method closes the window and terminates its processes and
    //  should be called by the user to kill the window when the
    //  process being monitored is through.
    // -------------------------------------------------------------------------------------------------------

    public void makeQuit() {
        runThread = false;
        this.hide();
        this.dispose();
    }

}



class ProgressPad extends Canvas {

    public void paint(Graphics g) {
        g.setColor(Color.black);
        g.drawRect(20,20,100,10);
    }

}