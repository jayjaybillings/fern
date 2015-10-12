package edu.utk.phys.fern;

public class QuizMaker {

    public static void main (String[] args) {

      // enable anti-aliased of text in Swing components
      
      System.setProperty("awt.useSystemAAFontSettings","on");
      System.setProperty("swing.aatext", "true");
	
      // Open first screen
      
      OpeningScreen os = new OpeningScreen(230,155," Quiz Editor");
      os.show();
      
    }
    
   
    //  Static method stringToInt to convert a string to an int

    static int stringToInt (String s) {
        Integer myInt=Integer.valueOf(s);     // String to Integer (object)
        return myInt.intValue();              // Return primitive int
    }

} 
