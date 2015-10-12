
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import javax.swing.*;

class EditOld extends JFrame implements ItemListener {
	
	
	
	public EditOld (int width, int height, String title, String text) {

        this.pack();
        this.setSize(width,height);
        this.setTitle(title);
		
		JPanel panel0 = new JPanel();
        panel0.setLayout(new GridLayout(1,1));
		JLabel first = new JLabel("Create New or Edit Old Quiz", JLabel.CENTER);
        panel0.add(first);
		
		JPanel panel1 = new JPanel();
        panel1.setLayout(new FlowLayout());
        //panel1.setFont(textFont);
        //panel1.setForeground(panelForeColor);

        JLabel sfL = new JLabel("File: ",JLabel.RIGHT);
        panel1.add(sfL);

		final JTextField sf = new JTextField(25);
		sf.setText("");
		panel1.add(sf);

//         JLabel asymptoticL = new JLabel("  Method ",JLabel.RIGHT);
//         panel1.add(asymptoticL);
// 		
// 		DefaultComboBoxModel model = new DefaultComboBoxModel();
// 		model.addElement("Asy");
// 		model.addElement("QSS");
// 		model.addElement("AsyMott");
// 		model.addElement("AsyOB");
// 		model.addElement("Explicit");
// 		model.addElement("F90Asy");
// 		model.addElement("Asy+PE");
// 		model.addElement("QSS+PE");
// 		JComboBox comboBox = new JComboBox(model);
// 		comboBox.setSelectedItem("Asy");
// 		panel1.add(comboBox);
// 				
				
		JPanel cboxPanel = new JPanel();
        cboxPanel.setLayout(new GridLayout(2,1));
		cboxPanel.add(panel0);
        cboxPanel.add(panel1);
		
		this.add(cboxPanel,"North");
		
		// Add Dismiss, Save Changes, Print, and Help buttons

        JPanel botPanel = new JPanel();

        
        JButton createButton = new JButton("Create");
        JButton editButton = new JButton("Edit");
		JButton helpButton = new JButton("Help");
		JButton dismissButton = new JButton("Cancel");
        //botPanel.add(label1);
        
        //botPanel.add(label2);
        botPanel.add(createButton);
        //botPanel.add(label3);
        //botPanel.add(printButton);
        //botPanel.add(label4);
        botPanel.add(editButton);
		botPanel.add(helpButton);
		botPanel.add(dismissButton);

        this.add("South", botPanel);
		
	}
	
	public void itemStateChanged(ItemEvent check) {
		
	}
	
	
}