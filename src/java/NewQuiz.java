import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import javax.swing.*;
import javax.swing.border.EmptyBorder;

class NewQuiz extends JFrame {
	
    public NewQuiz (int width, int height, String title, String text) {

	this.pack();
	this.setSize(width,height);
	this.setTitle(title);
		
	JPanel panelA = new JPanel();
	panelA.setLayout(new GridLayout(1,1));
	JLabel first = new JLabel("Create a New Quiz", JLabel.CENTER);
	panelA.add(first);
		
	JPanel panelB = new JPanel();
	panelB.setLayout(new FlowLayout());
	JLabel fileFieldL = new JLabel(" File: ", JLabel.RIGHT);
	panelB.add(fileFieldL);
	final JTextField fileField = new JTextField(22);
	fileField.setText("astro.json");
	panelB.add(fileField);
	
	JPanel panelC = new JPanel();
	panelC.setLayout(new FlowLayout());
	JLabel titleFieldL = new JLabel("Title: ", JLabel.RIGHT);
	panelC.add(titleFieldL);
	final JTextField titleField = new JTextField(22);
	titleField.setText("New Quiz");
	panelC.add(titleField);
	
	JPanel panelD = new JPanel();
	panelD.setLayout(new FlowLayout());
	JLabel qnumL = new JLabel("Answers ", JLabel.RIGHT);
	panelD.add(qnumL);		
	DefaultComboBoxModel model = new DefaultComboBoxModel();
	model.addElement("5");
	model.addElement("4");
	model.addElement("3");
	model.addElement("2");
	final JComboBox comboBox = new JComboBox(model);
	comboBox.setSelectedItem("5");
	panelD.add(comboBox);
	JLabel amplifyL = new JLabel("   Amplify? ", JLabel.RIGHT);
	panelD.add(amplifyL);	
	DefaultComboBoxModel model2 = new DefaultComboBoxModel();
	model2.addElement("True");
	model2.addElement("False");
	final JComboBox amplify = new JComboBox(model2);
	amplify.setSelectedItem("True");
	panelD.add(amplify);
					
	JPanel mainPanel = new JPanel();
	mainPanel.setLayout(new GridLayout(4,1,2,2));
	mainPanel.setBorder(new EmptyBorder(0,0,0,0)); // Top, left, bottom, right
	mainPanel.add(panelA);
	mainPanel.add(panelB);
	mainPanel.add(panelC);
	mainPanel.add(panelD);
	
	this.add(mainPanel,"North");
	
	// Add Dismiss, Save Changes, and Help buttons

	JPanel bottomPanel = new JPanel();
	bottomPanel.setLayout(new FlowLayout());
	bottomPanel.setBackground(MyColors.gray220);      
	JButton createButton = new JButton("Create");
	JButton helpButton = new JButton("Help");
	JButton dismissButton = new JButton("Cancel");
	bottomPanel.add(createButton);
	bottomPanel.add(helpButton);
	bottomPanel.add(dismissButton);
	
	this.add(bottomPanel, "South");
		
	// Handler for Create button
        
	createButton.addActionListener(new ActionListener() {
	  public void actionPerformed(ActionEvent ae){
  
	    UI.fileName = fileField.getText().trim();
	    UI.quizTitle = titleField.getText().trim();
	    UI.numberAnswers = QuizMaker.stringToInt(comboBox.getSelectedItem().toString());
	    UI.addAmplify = Boolean.parseBoolean(amplify.getSelectedItem().toString());
	    UI ui = new UI(575,650,titleField.getText().trim(),"");
	    ui.show();
	    hide();
	    dispose();
	  }
	});
				
	// Handler for Dismiss button

	dismissButton.addActionListener(new ActionListener() {
	    public void actionPerformed(ActionEvent ae){
		  hide();
		  dispose();
	    }
	});
	
	// Window closing handler

        this.addWindowListener(new WindowAdapter() {
	    public void windowClosing(WindowEvent e) {
		hide();
		dispose();
	    }
	 });
		
	}
	
}