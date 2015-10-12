
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import javax.swing.*;
import javax.swing.border.EmptyBorder;

class OpeningScreen extends JFrame {
	
	public OpeningScreen (int width, int height, String title) {

        this.pack();
        this.setSize(width,height);
        this.setTitle(title);
		
		JButton createButton = new JButton("New Quiz");
        JButton editButton = new JButton("Edit Old Quiz");
		JButton helpButton = new JButton("Help");
		JButton dismissButton = new JButton("Cancel");
 				
		JPanel panelA = new JPanel();
		panelA.setLayout(new FlowLayout());
		panelA.add(createButton);
		
		JPanel panelB = new JPanel();
		panelB.setLayout(new FlowLayout());
		panelB.add(editButton);
		
		
		JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new GridLayout(2,1));

        mainPanel.add(panelA);
		mainPanel.add(panelB);
		
		//mainPanel.setBorder(BorderFactory.createMatteBorder( 4, 0, 0, 0, mainPanel.getBackground() ) ); 
		
		mainPanel.setBorder(new EmptyBorder(10,0,10,0)); // Top, left, bottom, right
		
		this.add(mainPanel,"North");
		
		// Add Dismiss and Help buttons

        JPanel bottomPanel = new JPanel();
		bottomPanel.setLayout(new FlowLayout());
		bottomPanel.setBackground(MyColors.gray220);

		bottomPanel.add(helpButton);
		bottomPanel.add(dismissButton);

        this.add(bottomPanel, "South");
		
		// Create button handler
        
        createButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
				NewQuiz newQuiz = new NewQuiz(350,220,"NewQuiz","");
				newQuiz.show();
				hide();
				dispose();
            }
        });
		
		// Edit button handler.
        
        editButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
        
             EditOld oldQuiz = new EditOld(400,300,"OldQuiz","");
                    oldQuiz.show();
					hide();
					dispose();
            }
        });
		
		// Dismiss button handler

		dismissButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent ae){
				hide();
				dispose();
			}
		});
		
		// Window close handler

        this.addWindowListener(new WindowAdapter() {
           public void windowClosing(WindowEvent e) {
              hide();
              dispose();
           }
        });
		
	}
	
}