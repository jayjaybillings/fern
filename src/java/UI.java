
//  Class UI to enter test questions and answers

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import javax.swing.*;
import javax.swing.border.EmptyBorder;

class UI extends JFrame {

    static String fileName;
    static int numberAnswers;
    static boolean addAmplify;
    static String quizTitle;
    static int maxQuestions = 100;
    
    private static final String tab = "    ";
    private static final String tab2 = tab + tab;
    private static final String tab3 = tab + tab + tab;
    private static final String responseDataOpen = "{\"responseData\":";
    private static final String responseDataClose = "}";
    private static final String questionsOpen = tab + "{\"questions\":[";
    private static final String questionsClose = tab + "]\n" + tab + "}";
    
    private int currentCorrect;
    private int numberQuestions = 1;
    public static String q[] = new String [maxQuestions];
    public static String a1[] = new String [maxQuestions];
    public static String a2[] = new String [maxQuestions];
    public static String a3[] = new String [maxQuestions];
    public static String a4[] = new String [maxQuestions];
    public static String a5[] = new String [maxQuestions];
    public static String coran[] = new String [maxQuestions];
    public static String ampRemark[] = new String [maxQuestions];
    
    static FileOutputStream to2;
    static PrintWriter toY;
    
    static boolean helpWindowOpen = false;
    GenericHelpDialog hf;

    Color disablebgColor = new Color(230,230,230);
    Color disablefgColor = new Color(180,180,180);

    private int vtextSize = 4;
    private int htextSize = 40;
    
    JTextField questionNumberField;
    JTextArea questionField;
    JTextArea a1Field;
    JTextArea a2Field;
    JTextArea a3Field;
    JTextArea a4Field;
    JTextArea a5Field;
    JTextArea amp;
    DefaultComboBoxModel model1;
    JComboBox correct;
    JLabel questionFieldL;
    

    public UI (int width, int height, String title, String text) {

	this.pack();
	this.setSize(width,height);
	this.setTitle(title);

// 	JPanel panelA = new JPanel();
// 	panelA.setLayout(new FlowLayout());
// 	JLabel questionNumberFieldL = new JLabel("Question number  ");
// 	questionNumberField = new JTextField(2);
// 	questionNumberField.setEditable(false);
// 	questionNumberField.setText(Integer.toString(numberQuestions));
// 	panelA.add(questionNumberFieldL);
// 	panelA.add(questionNumberField);
	
	JPanel panelB = new JPanel();
	panelB.setLayout(new FlowLayout());
	questionFieldL = new JLabel("Question "+numberQuestions +" ", 
	  SwingConstants.RIGHT);
	questionField = new JTextArea(vtextSize, htextSize);
	questionField.setLineWrap(true);
	questionField.setWrapStyleWord(true);
	questionField.setText("");
	panelB.add(questionFieldL);
	panelB.add(questionField);
	
	JPanel panelC = new JPanel();
	panelC.setLayout(new FlowLayout());
	JLabel a1L = new JLabel(" Answer A  ", SwingConstants.RIGHT);
	a1Field = new JTextArea(vtextSize, htextSize);
	a1Field.setLineWrap(true);
	a1Field.setWrapStyleWord(true);
	a1Field.setText("");
	panelC.add(a1L);
	panelC.add(a1Field);
	
	JPanel panelD = new JPanel();
	panelD.setLayout(new FlowLayout());
	JLabel a2L = new JLabel(" Answer B  ", SwingConstants.RIGHT);
	a2Field = new JTextArea(vtextSize, htextSize);
	a2Field.setLineWrap(true);
	a2Field.setWrapStyleWord(true);
	a2Field.setText("");
	panelD.add(a2L);
	panelD.add(a2Field);
	
	JPanel panelE = new JPanel();
	panelE.setLayout(new FlowLayout());
	JLabel a3L = new JLabel(" Answer C  ", SwingConstants.RIGHT);
	a3Field = new JTextArea(vtextSize, htextSize);
	if(numberAnswers < 3){
	  a3Field.setEditable(false);
	  a3Field.setBackground(disablebgColor);
	  a3L.setForeground(disablefgColor);
	} 
	a3Field.setLineWrap(true);
	a3Field.setWrapStyleWord(true);
	a3Field.setText("");
	panelE.add(a3L);
	panelE.add(a3Field);
	
	JPanel panelF = new JPanel();
	panelF.setLayout(new FlowLayout());
	JLabel a4L = new JLabel(" Answer D  ", SwingConstants.RIGHT);
	a4Field = new JTextArea(vtextSize, htextSize);
	if(numberAnswers < 4){
	  a4Field.setEditable(false);
	  a4Field.setBackground(disablebgColor);
	  a4L.setForeground(disablefgColor);
	}
	a4Field.setLineWrap(true);
	a4Field.setWrapStyleWord(true);
	a4Field.setText("");
	panelF.add(a4L);
	panelF.add(a4Field);
	
	JPanel panelG = new JPanel();
	panelG.setLayout(new FlowLayout());
	JLabel a5L = new JLabel(" Answer E  ", SwingConstants.RIGHT);
	a5Field = new JTextArea(vtextSize, htextSize);
	if(numberAnswers < 5){
	  a5Field.setEditable(false);
	  a5Field.setBackground(disablebgColor);
	  a5L.setForeground(disablefgColor);
	}
	a5Field.setLineWrap(true);
	a5Field.setWrapStyleWord(true);
	a5Field.setText("");
	panelG.add(a5L);
	panelG.add(a5Field);
	
	JPanel panelH = new JPanel();
	panelH.setLayout(new FlowLayout());
	JLabel ampL = new JLabel("    Amplify  ", SwingConstants.RIGHT);	
	amp = new JTextArea(vtextSize, htextSize);
	if(!addAmplify){
	  amp.setEditable(false);
	  amp.setBackground(disablebgColor);
	  ampL.setForeground(disablefgColor);
	}
	amp.setLineWrap(true);
	amp.setWrapStyleWord(true);
	amp.setText("");
	panelH.add(ampL);
	panelH.add(amp);
	
	JPanel panelI = new JPanel();
	panelI.setLayout(new FlowLayout(FlowLayout.CENTER, 0,10));
	JLabel correctL = new JLabel("Correct answer ");
	model1 = new DefaultComboBoxModel();
	model1.addElement("A");
	model1.addElement("B");
	model1.addElement("C");
	model1.addElement("D");
	model1.addElement("E");
	model1.addElement("");
	correct = new JComboBox(model1);
	correct.setSelectedItem("");
	panelI.add(correctL);
	panelI.add(correct);
	JLabel spacerL = new JLabel("         ");
	panelI.add(spacerL);
	JButton newQuestion = new JButton("New Question");
	panelI.add(newQuestion);
	
	// Main panel to hold all the widgets

	JPanel mainPanel = new JPanel();
	mainPanel.setLayout(new GridLayout(8,1,0,0));   // Rows, columns, dx, dy
	mainPanel.setBorder(new EmptyBorder(15,0,0,0)); // Top, left, bottom, right
	//mainPanel.add(panelA);
	mainPanel.add(panelB);
	mainPanel.add(panelC);
	mainPanel.add(panelD);
	mainPanel.add(panelE);
	mainPanel.add(panelF);
	mainPanel.add(panelG);
	mainPanel.add(panelH);
	mainPanel.add(panelI);

	// Add the main to frame layout

	this.add(mainPanel,"North");
		
	// Add the bottom panel
	
	JPanel bottomPanel = new JPanel();
	bottomPanel.setBackground(MyColors.gray204);
	JButton cancelButton = new JButton("Cancel");
	JButton saveFileButton = new JButton("Save Quiz");
	JButton helpMeButton = new JButton("  Help  ");
	bottomPanel.add(saveFileButton);
	bottomPanel.add(helpMeButton);
	bottomPanel.add(cancelButton);

	this.add("South", bottomPanel);
		

	// Add inner class event handler for Dismiss button

	cancelButton.addActionListener(new ActionListener() {
	    public void actionPerformed(ActionEvent ae){
	      hide();
	      dispose();
	    }
	});
		
	// Event handler for new question button
		
	newQuestion.addActionListener(new ActionListener() {
	    public void actionPerformed(ActionEvent ae){
	    
	    
	      if(correct.getSelectedItem().toString().toLowerCase().equals("")) {
	      
		  new MakeWarning (100, 100, 180, 140, "ERROR!", 
		      "You must specify the correct answer before you can save this question.", 
		      UI.this
		  );

	      } else {

		  try {
		      
		      saveQuestion(numberQuestions-1);
		      numberQuestions ++;
		      clearAll();

		  } catch(NumberFormatException e) {

		  }
		  
	      }
	    }
	}); 


	// Inner class event handler for Save file button

	saveFileButton.addActionListener(new ActionListener() {
	    public void actionPerformed(ActionEvent ae){

		try {       // Catch NumberFormatExceptions and warn user
					
		    writeQuestions();

		} catch(NumberFormatException e) {

		}
	    }
	}); 


	// Help button handler
	
	helpMeButton.addActionListener(new ActionListener() {
	    public void actionPerformed(ActionEvent ae){
	
		if(helpWindowOpen && hf != null) {
		    hf.toFront();
		} else {
		    hf = new GenericHelpDialog(UI.makeHelpString(),
			" Help File", 200, 300, 100, 100, UI.this);
		    hf.show();
		    helpWindowOpen = true;
		}
	    }
	});

	// Add window closing button (inner class)

	this.addWindowListener(new WindowAdapter() {
	  public void windowClosing(WindowEvent e) {
	      hide();
	      dispose();
	  }
	});
			
  }  

  
  //  Static method to generate string for Help file

  static String makeHelpString() {

    String s;
    s="Fill in the question, five possible answers A-E, an optional remark";
    s+=" amplifying on the correct answer, and indicate the correct answer.";
    s+=" Click New Question once you are ready to create the next question.";
    s+=" Click Save Quiz to save all questions to a file.";
	    
    return s;

    }

    // Method to write the quiz to the output file

    public void writeQuestions(){

    System.out.println("Writing output file");

    try {

      to2 = new FileOutputStream(fileName);
      toY = new PrintWriter(to2);
      
      toY.println(responseDataOpen);
      toY.println(questionsOpen);
      
      for(int j=0; j<numberQuestions-1; j++){
	      writeOneQuestion(j);
      }
      toY.println(questionsClose);
      toY.println(responseDataClose);
	    
	    
    } catch (IOException e) {;}

    toY.flush();	
    toY.close();

    try {
      to2.close();	
    } catch (IOException e) {
	    
    }
  }

  // Method to write a single question to the output stream

  public void writeOneQuestion(int qnum){

      toY.println();
      toY.println(tab2 + "{");
      toY.println(tab3 + "\"q\": " + "\"" + q[qnum]+"?\",");
      toY.println(tab3 + "\"a\": " + "\"" + a1[qnum]+"\",");
      toY.println(tab3 + "\"b\": " + "\"" + a2[qnum]+"\",");
      toY.println(tab3 + "\"c\": " + "\"" + a3[qnum]+"\",");
      toY.println(tab3 + "\"d\": " + "\"" + a4[qnum]+"\",");
      toY.println(tab3 + "\"e\": " + "\"" + a5[qnum]+"\",");
      toY.println(tab3 + "\"coran\": " + "\"" + coran[qnum]+"\",");
      toY.println(tab3 + "\"amp\": " + "\"" + ampRemark[qnum]+"\".");
      
      toY.println(tab2 + "},");
	  
  }

  // Method to save current question to arrays

  public void saveQuestion(int qnum){

      q[qnum] = questionField.getText().trim();
      a1[qnum] = a1Field.getText().trim();
      a2[qnum] = a2Field.getText().trim();
      a3[qnum] = a3Field.getText().trim();
      a4[qnum] = a4Field.getText().trim();
      a5[qnum] = a5Field.getText().trim();
      coran[qnum] = correct.getSelectedItem().toString().toLowerCase();
      ampRemark[qnum] = amp.getText().trim();
      
  }

  // Method to clear fields for new question

  public void clearAll(){

      questionField.setText("");
      a1Field.setText("");
      a2Field.setText("");
      a3Field.setText("");
      a4Field.setText("");
      a5Field.setText("");
      correct.setSelectedItem("");
      amp.setText("");
      questionFieldL.setText("Question "+numberQuestions+"  ");
	  
  }
	
}