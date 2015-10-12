// ------------------------------------------------------------------------------------------------------------
//  Class ParamSetup to set up basic parameters for calculation.
//  Launches window in which these parameters can be specified. (Window
//  is opened from button in instance of SegreFrame.)
// ------------------------------------------------------------------------------------------------------------


import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;

class ParamSetup extends Frame implements ItemListener {

    static boolean helpWindowOpen = false;
    GenericHelpFrame hf = new GenericHelpFrame("","",0,0,0,0,0,0);

    static final double LOG10 = 0.434294482;

    Font titleFont = new java.awt.Font("SanSerif", Font.BOLD, 12);
    FontMetrics titleFontMetrics = getFontMetrics(titleFont);
    Font buttonFont = new java.awt.Font("SanSerif", Font.BOLD, 11);
    FontMetrics buttonFontMetrics = getFontMetrics(buttonFont);
    Font textFont = new java.awt.Font("SanSerif", Font.PLAIN, 12);
    FontMetrics textFontMetrics = getFontMetrics(textFont);

    Color panelForeColor = Color.black;
    Color panelBackColor = Color.white;
    Color disablebgColor = new Color(230,230,230);
    Color disablefgColor = new Color(153,153,153);
    Color framebgColor = new Color(230,230,230);

    Panel panel4, panel5;
    TextField profile, rho, T9;
    Label profileL, rhoL, T9L;

    CheckboxGroup cbg = new CheckboxGroup();
    final Checkbox [] checkBox = new Checkbox[2];

    public ParamSetup (int width, int height, String title, String text) {

        this.pack();
        this.setSize(width,height);
        this.setTitle(title);
        this.setBackground(framebgColor);

        String temp;

        Panel panel0 = new Panel();
        panel0.setLayout(new GridLayout(1,1));
        panel0.setFont(titleFont);
        panel0.setForeground(panelForeColor);

        Label first = new Label("  Integration Parameters",Label.LEFT);
        panel0.add(first);

        Panel panel1 = new Panel();
        panel1.setLayout(new FlowLayout());
        panel1.setFont(textFont);
        panel1.setForeground(panelForeColor);

        Label sfL = new Label("Precision",Label.RIGHT);
        panel1.add(sfL);

		final TextField sf = new TextField(6);
		sf.setFont(textFont);
		if (StochasticElements.stochasticFactor != 0) {
			temp = Double.toString(StochasticElements.stochasticFactor);
		} else {
			temp = "";
		}
		sf.setText(temp);
		sf.setBackground(panelBackColor);
		panel1.add(sf);

        Label asymptoticL = new Label("Integration method",Label.RIGHT);
        panel1.add(asymptoticL);

        final Choice asymptotic = new Choice();
        asymptotic.setFont(textFont);
        asymptotic.setBackground(panelBackColor);
        asymptotic.addItem("Asy");
        asymptotic.addItem("QSS");
        asymptotic.addItem("AsyMott");
        asymptotic.addItem("AsyOB");
        asymptotic.addItem("Explicit");
        asymptotic.addItem("F90Asy");
        asymptotic.addItem("Asy+PE");
        asymptotic.addItem("QSS+PE");
        if(!StochasticElements.integrateWithJava){
            asymptotic.select(5);
        } else if (StochasticElements.doAsymptotic && StochasticElements.imposeEquil) {
        	asymptotic.select(6);
        } else if (StochasticElements.doSS && StochasticElements.imposeEquil) {
        	asymptotic.select(7);
        } else if (StochasticElements.doSS) {
           asymptotic.select(1);
        } else if (StochasticElements.doAsymptotic && ! StochasticElements.asyPC){
            asymptotic.select(0);
        } else if (StochasticElements.doAsymptotic && StochasticElements.asyPC
            && StochasticElements.isMott){
            asymptotic.select(2);
        } else if (StochasticElements.doAsymptotic && StochasticElements.asyPC
            && ! StochasticElements.isMott){
            asymptotic.select(3);
        } else {
            asymptotic.select(4);
        }
        
        panel1.add(asymptotic);

        Label massTolL = new Label("dX",Label.RIGHT);
        panel1.add(massTolL);

        final TextField  massTol = new TextField(5);
        massTol.setFont(textFont);
        temp = Double.toString(StochasticElements.massTol);
        massTol.setText(temp);
        massTol.setBackground(panelBackColor);
        panel1.add(massTol);

        Panel panel2 = new Panel();
        panel2.setLayout(new FlowLayout());
        panel2.setFont(textFont);
        panel2.setForeground(panelForeColor);
        Label logtminL = new Label("Log10 Start time (s)",Label.RIGHT);
        panel2.add(logtminL);

        final TextField  logtmin = new TextField(8);
        logtmin.setFont(textFont);
        temp = Double.toString(StochasticElements.logtmin);
        logtmin.setText(temp);
        logtmin.setBackground(panelBackColor);
        panel2.add(logtmin);

        Label logtmaxL = new Label("Log10 End time (s)",Label.RIGHT);
        panel2.add(logtmaxL);

        final TextField  logtmax = new TextField(8);
        logtmax.setFont(textFont);
        temp = Double.toString(StochasticElements.logtmax);
        logtmax.setText(temp);
        logtmax.setBackground(panelBackColor);
        panel2.add(logtmax);
        
        // --- Panel for equilibrium quantities
        
        Panel panelEq = new Panel();
        panelEq.setLayout(new FlowLayout());
        panelEq.setFont(textFont);
        panelEq.setForeground(panelForeColor);
		
		Label LtrackEq = new Label("Track Equil ",Label.RIGHT);
        panelEq.add(LtrackEq);

        final Choice trackEq = new Choice();
        trackEq.setFont(textFont);
        trackEq.setBackground(panelBackColor);
        trackEq.addItem("No");
        trackEq.addItem("Yes");
        if(StochasticElements.equilibrate) {
           trackEq.select(1);
        } else {
            trackEq.select(0);
        }
        panelEq.add(trackEq);

        Label eqTimeL = new Label("Equil time",Label.RIGHT);
        panelEq.add(eqTimeL);

		final TextField eqTime = new TextField(6);
		eqTime.setFont(textFont);
		if (StochasticElements.equilibrateTime != 0) {
			temp = Double.toString(StochasticElements.equilibrateTime);
		} else {
			temp = "";
		}
		eqTime.setText(temp);
		eqTime.setBackground(panelBackColor);
		panelEq.add(eqTime);
		
		Label eqTolL = new Label("Equil tolerance",Label.RIGHT);
        panelEq.add(eqTolL);

		final TextField eqTol = new TextField(6);
		eqTol.setFont(textFont);
		if (StochasticElements.equiTol != 0) {
			temp = Double.toString(StochasticElements.equiTol);
		} else {
			temp = "";
		}
		eqTol.setText(temp);
		eqTol.setBackground(panelBackColor);
		panelEq.add(eqTol);
		
		// -----------------
		
		
        Panel panel2B = new Panel();
        panel2B.setLayout(new FlowLayout());
        panel2B.setFont(textFont);
        panel2B.setForeground(panelForeColor);

        Label lpFormatL = new Label("Lineplot",Label.RIGHT);
        panel2B.add(lpFormatL);

        final Choice lpFormat = new Choice();
        lpFormat.setFont(textFont);
        lpFormat.setBackground(panelBackColor);
        lpFormat.addItem("Short");
        lpFormat.addItem("Tall");
        if(StochasticElements.longFormat) {
            lpFormat.select(1);
        } else {
            lpFormat.select(0);
        }
        panel2B.add(lpFormat);

        // Population 2D animation color maps.  See the class MyColors for definitions
        
        Label popCML = new Label("Pop CM",Label.RIGHT);
        panel2B.add(popCML);

        final Choice popCM = new Choice();
        popCM.setFont(textFont);
        popCM.setBackground(panelBackColor);
        popCM.addItem("guidry2");
        popCM.addItem("guidry");
        popCM.addItem("hot");
        popCM.addItem("bluehot");
        popCM.addItem("greyscale");
        popCM.addItem("caleblack");
        popCM.addItem("calewhite");
        popCM.addItem("cardall");

        popCM.select(StochasticElements.popColorMap);
        panel2B.add(popCM);
        
        // Flux 2D animation color maps.  See the class MyColors for definitions
        
        Label fluxCML = new Label("Flux CM",Label.RIGHT);
        panel2B.add(fluxCML);

        final Choice fluxCM = new Choice();
        fluxCM.setFont(textFont);
        fluxCM.setBackground(panelBackColor);
        fluxCM.addItem("guidry2");
        fluxCM.addItem("guidry");
        fluxCM.addItem("hot");
        fluxCM.addItem("bluehot");
        fluxCM.addItem("greyscale");
        fluxCM.addItem("caleblack");
        fluxCM.addItem("calewhite");
        fluxCM.addItem("cardall");

        fluxCM.select(StochasticElements.fluxColorMap);
        if(StochasticElements.doFluxPlots) {
            fluxCM.enable();
        } else {
            fluxCM.disable();
        }
        panel2B.add(fluxCM);

        Panel panel3 = new Panel();
        panel3.setLayout(new GridLayout(1,1));
        panel3.setFont(titleFont);
        panel3.setForeground(panelForeColor);

        Label trho = new Label("  Hydrodynamic Variables",Label.LEFT);
        panel3.add(trho);

        // Create two checkboxes

        checkBox[0] = new Checkbox("Specify profile");
        checkBox[1] = new Checkbox("Constant:");

        // Make them part of a checkbox group (exclusive radio buttons)

        checkBox[0].setCheckboxGroup(cbg);
        checkBox[1].setCheckboxGroup(cbg);

        // Add itemListeners to listen for checkbox events.  These events
        // will be processed by the method itemStateChanged

        checkBox[0].addItemListener(this);
        checkBox[1].addItemListener(this);

        panel4 = new Panel();
        panel4.setLayout(new FlowLayout());
        panel4.setFont(textFont);
        panel4.setForeground(panelForeColor);

        panel4.add(checkBox[0]);

        profileL = new Label("File",Label.RIGHT);
        panel4.add(profileL);

        profile = new TextField(29);
        profile.setFont(textFont);
        temp = StochasticElements.profileFileName;
        profile.setText(temp);
        panel4.add(profile);

        panel5 = new Panel();
        panel5.setLayout(new FlowLayout());
        panel5.setFont(textFont);
        panel5.setForeground(panelForeColor);

        panel5.add(checkBox[1]);

        T9L = new Label("T9",Label.RIGHT);
        panel5.add(T9L);

        T9 = new TextField(7);
        T9.setFont(textFont);
        if(StochasticElements.T9 != 0) {
            temp = Double.toString(StochasticElements.T9);
        } else {temp="";}
        T9.setText(temp);
        T9.setBackground(panelBackColor);
        panel5.add(T9);

        rhoL = new Label("rho(cgs)",Label.RIGHT);
        panel5.add(rhoL);

        rho = new TextField(7);
        rho.setFont(textFont);
        if(StochasticElements.rho != 0) {
            temp = Double.toString(StochasticElements.rho);
        } else {temp="";}
        rho.setText(temp);
        rho.setBackground(panelBackColor);
        panel5.add(rho);

        Label YeL = new Label("Ye",Label.RIGHT);
        panel5.add(YeL);
        final TextField  Ye = new TextField(4);
        Ye.setFont(textFont);
        if(StochasticElements.Ye != 0) {
            temp = Double.toString(StochasticElements.Ye);
        } else {temp="";}
        Ye.setText(temp);
        Ye.setBackground(panelBackColor);
        panel5.add(Ye);

        // Set the current values of the temperature/density fields

        if(StochasticElements.constantHydro) {         // Constant T, rho
            profileL.setForeground(disablefgColor);
            profile.disable();
            profile.setBackground(disablebgColor);
            rhoL.setForeground(panelForeColor);
            rho.enable();
            rho.setBackground(panelBackColor);
            T9.setForeground(panelForeColor);
            T9.enable();
            T9.setBackground(panelBackColor);
            cbg.setSelectedCheckbox(checkBox[1]);
        } else {                                      // T, rho time profile
            profileL.setForeground(panelForeColor);
            profile.enable();
            profile.setBackground(panelBackColor);
            rhoL.setForeground(disablefgColor);
            rho.disable();
            rho.setBackground(disablebgColor);
            T9L.setForeground(disablefgColor);
            T9.disable();
            T9.setBackground(disablebgColor);
            cbg.setSelectedCheckbox(checkBox[0]);
        }

		

        Panel panel6a = new Panel();
        panel6a.setLayout(new GridLayout(1,1));
        panel6a.setFont(titleFont);
        panel6a.setForeground(panelForeColor);

        Label lab6a = new Label("  Plot Control Parameters",Label.LEFT);
        panel6a.add(lab6a);

        Panel panel6 = new Panel();
        panel6.setLayout(new FlowLayout());
        panel6.setFont(textFont);
        panel6.setForeground(panelForeColor);

        Label LtminPlot = new Label("logx-",Label.RIGHT);
        panel6.add(LtminPlot);

        final TextField  logtminPlot = new TextField(6);
        logtminPlot.setFont(textFont);
        logtminPlot.setText(Double.toString(StochasticElements.logtminPlot));
        logtminPlot.setBackground(panelBackColor);
        panel6.add(logtminPlot);

        Label LtmaxPlot = new Label("logx+",Label.RIGHT);
        panel6.add(LtmaxPlot);

        final TextField  logtmaxPlot = new TextField(6);
        logtmaxPlot.setFont(textFont);
        logtmaxPlot.setText(Double.toString(StochasticElements.logtmaxPlot));
        logtmaxPlot.setBackground(panelBackColor);
        panel6.add(logtmaxPlot);

        Label LXmin = new Label("logy-",Label.RIGHT);
        panel6.add(LXmin);

        final TextField  XminPlot = new TextField(4);
        XminPlot.setFont(textFont);
        temp = Double.toString(StochasticElements.yminPlot );
        XminPlot.setText(temp);
        XminPlot.setBackground(panelBackColor);
        panel6.add(XminPlot);

        Label LXmax = new Label("logy+",Label.RIGHT);
        panel6.add(LXmax);

        final TextField  XmaxPlot = new TextField(4);
        XmaxPlot.setFont(textFont);
        temp = Double.toString(StochasticElements.ymaxPlot);
        XmaxPlot.setText(temp);
        XmaxPlot.setBackground(panelBackColor);
        panel6.add(XmaxPlot);
		
        Panel panel8 = new Panel();
        panel8.setLayout(new FlowLayout());
        panel8.setFont(textFont);
        panel8.setForeground(panelForeColor);

        Label Lxtics = new Label("x tics",Label.RIGHT);
        panel8.add(Lxtics);

        final TextField  xtics = new TextField(3);
        xtics.setFont(textFont);
        if(StochasticElements.xtics != 0) {
            temp = Integer.toString(StochasticElements.xtics);
        } else {temp="";}
        xtics.setText(temp);
        xtics.setBackground(panelBackColor);
        panel8.add(xtics);

        Label Lytics = new Label("y tics",Label.RIGHT);
        panel8.add(Lytics);

        final TextField  ytics = new TextField(3);
        ytics.setFont(textFont);
        if(StochasticElements.ytics != 0) {
            temp = Integer.toString(StochasticElements.ytics);
        } else {temp="";}
        ytics.setText(temp);
        ytics.setBackground(panelBackColor);
        panel8.add(ytics);

        Label LmaxCurves = new Label("Isotopes",Label.RIGHT);
        panel8.add(LmaxCurves);

        final TextField  maxCurves = new TextField(4);
        maxCurves.setFont(textFont);
        if(StochasticElements.maxToPlot != 0) {
            temp = Integer.toString(StochasticElements.maxToPlot);
        } else {temp="";}
        maxCurves.setText(temp);
        maxCurves.setBackground(panelBackColor);
        panel8.add(maxCurves);

        Label energyL = new Label("E/dE ",Label.RIGHT);
        panel8.add(energyL);

        final Choice energy = new Choice();
        energy.setFont(textFont);
        energy.setBackground(panelBackColor);
        energy.addItem("E");
        energy.addItem("dE");
        energy.addItem("None");
        if(!StochasticElements.plotEnergy) {
            energy.select(2);
        } else if(StochasticElements.plotdE){
            energy.select(1);
        } else {
            energy.select(0);
        }

        panel8.add(energy);

        Panel panel9 = new Panel();
        panel9.setLayout(new FlowLayout());
        panel9.setFont(textFont);
        panel9.setForeground(panelForeColor);

        Label LplotY = new Label("X/Y ",Label.RIGHT);
        panel9.add(LplotY);

        final Choice plotY = new Choice();
        plotY.setFont(textFont);
        plotY.setBackground(panelBackColor);
        plotY.addItem("X");
        plotY.addItem("Y");
        if(StochasticElements.plotY) {
           plotY.select(1);
        } else {
            plotY.select(0);
        }
        panel9.add(plotY);


        Label LlinesOnly = new Label("Lines/Symbols",Label.RIGHT);
        panel9.add(LlinesOnly);

        final Choice linesOnly = new Choice();
        linesOnly.setFont(textFont);
        linesOnly.setBackground(panelBackColor);
        linesOnly.addItem("Lines");
        linesOnly.addItem("Symbols");
        if(StochasticElements.linesOnly) {
           linesOnly.select(0);
        } else {
            linesOnly.select(1);
        }
        panel9.add(linesOnly);


        Label LblackOnly = new Label("Color/BW",Label.RIGHT);
        panel9.add(LblackOnly);

        final Choice blackOnly = new Choice();
        blackOnly.setFont(textFont);
        blackOnly.setBackground(panelBackColor);
        blackOnly.addItem("Color");
        blackOnly.addItem("B/W");
        if(StochasticElements.blackOnly) {
           blackOnly.select(1);
        } else {
            blackOnly.select(0);
        }
        panel9.add(blackOnly);


        Panel panel10 = new Panel();
        panel10.setLayout(new FlowLayout());
        panel10.setFont(textFont);
        panel10.setForeground(panelForeColor);

        Label nintervalsL = new Label("Steps",Label.RIGHT);
        panel10.add(nintervalsL);

        final TextField  nintervals = new TextField(3);
        nintervals.setFont(textFont);
        if(StochasticElements.nintervals != 0) {
            temp = Integer.toString(StochasticElements.nintervals);
        } else {temp="";}
        nintervals.setText(temp);
        nintervals.setBackground(panelBackColor);
        panel10.add(nintervals);

        Label LminContour = new Label("MinCon",Label.RIGHT);
        panel10.add(LminContour);

        final TextField  minContour = new TextField(6);
        minContour.setFont(textFont);
        if(StochasticElements.minLogContour != 0) {
            temp = Double.toString(StochasticElements.minLogContour);
        } else {temp="";}
        minContour.setText(temp);
        minContour.setBackground(panelBackColor);
        panel10.add(minContour);

        Label Lwrite3D = new Label("3D",Label.RIGHT);
        panel10.add(Lwrite3D);

        final Choice write3D = new Choice();
        write3D.setFont(textFont);
        write3D.setBackground(panelBackColor);
        write3D.addItem("No");
        write3D.addItem("Yes");
        if(StochasticElements.write3DOutput) {
           write3D.select(1);
        } else {
            write3D.select(0);
        }
        panel10.add(write3D);

        Label LalphaOnly = new Label("AlphaOnly",Label.RIGHT);
        panel10.add(LalphaOnly);
		
        final Choice alphaOnly = new Choice();
        alphaOnly.setFont(textFont);
        alphaOnly.setBackground(panelBackColor);
        alphaOnly.addItem("No");
        alphaOnly.addItem("Yes");
        if(StochasticElements.tripleAlphaOnly) {
           alphaOnly.select(1);
        } else {
            alphaOnly.select(0);
        }
        panel10.add(alphaOnly);

        Panel panel11 = new Panel();
        panel11.setLayout(new FlowLayout());
        panel11.setFont(textFont);
        panel11.setForeground(panelForeColor);

        Label Lxdeci = new Label("x deci",Label.RIGHT);
        panel11.add(Lxdeci);

        final TextField  xdeci = new TextField(2);
        xdeci.setFont(textFont);
        temp = Integer.toString(StochasticElements.xdeci);
        xdeci.setText(temp);
        xdeci.setBackground(panelBackColor);
        panel11.add(xdeci);

        Label Lydeci = new Label("y deci",Label.RIGHT);
        panel11.add(Lydeci);

        final TextField  ydeci = new TextField(2);
        ydeci.setFont(textFont);
        temp = Integer.toString(StochasticElements.ydeci);
        ydeci.setText(temp);
        ydeci.setBackground(panelBackColor);
        panel11.add(ydeci);

        Label LYmin = new Label("Ymin",Label.RIGHT);
        panel11.add(LYmin);

        final TextField  Ymin = new TextField(6);
        Ymin.setFont(textFont);
        temp = Double.toString(StochasticElements.Ymin);
        Ymin.setText(temp);
        Ymin.setBackground(panelBackColor);
        panel11.add(Ymin);

        Label LrenormX = new Label("normX",Label.RIGHT);
        panel11.add(LrenormX);

        final Choice renormX = new Choice();
        renormX.setFont(textFont);
        renormX.setBackground(panelBackColor);
        renormX.addItem("No");
        renormX.addItem("Yes");
        if(StochasticElements.renormalizeMassFractions) {
           renormX.select(1);
        } else {
            renormX.select(0);
        }
        panel11.add(renormX);

        Panel panel12 = new Panel();
        panel12.setLayout(new FlowLayout());
        panel12.setFont(textFont);
        panel12.setForeground(panelForeColor);
		
        final TextArea commy = new TextArea("",1, 55, TextArea.SCROLLBARS_NONE);
        commy.setFont(textFont);
        temp = StochasticElements.myComment;
        commy.setText(temp);
        commy.setBackground(panelBackColor);
        panel12.add(commy);


// Remove the following options for now

/*
        Panel panelA = new Panel();
        panelA.setLayout(new GridLayout(1,1));
        panelA.setFont(titleFont);
        panelA.setForeground(panelForeColor);

        Label last = new Label("  Integration Isotope Ranges",Label.LEFT);
        panelA.add(last);

        Panel panel7 = new Panel();
        panel7.setLayout(new FlowLayout());
        panel7.setFont(textFont);
        panel7.setForeground(panelForeColor);

        Label zmaxL = new Label("Zmax",Label.RIGHT);
        panel7.add(zmaxL);

        final TextField  zmax = new TextField(3);
        zmax.setFont(textFont);
        if(StochasticElements.pmax != 0) {
            temp = Integer.toString(StochasticElements.pmax);
        } else {temp="";}
        zmax.setText(temp);
        zmax.setBackground(panelBackColor);
        panel7.add(zmax);

        Label nmaxL = new Label("Nmax",Label.RIGHT);
        panel7.add(nmaxL);

        final TextField  nmax = new TextField(3);
        nmax.setFont(textFont);
        if(StochasticElements.nmax != 0) {
            temp = Integer.toString(StochasticElements.nmax);
        } else {temp="";}
        nmax.setText(temp);
        nmax.setBackground(panelBackColor);
        panel7.add(nmax);

        Label zminL = new Label("Zmin",Label.RIGHT);
        panel7.add(zminL);

        final TextField  zmin = new TextField(3);
        zmin.setFont(textFont);
        if(StochasticElements.pmin != 0) {
            temp = Integer.toString(StochasticElements.pmin);
        } else {temp="";}
        zmin.setText(temp);
        zmin.setBackground(panelBackColor);
        panel7.add(zmin);

*/

        // Panel to hold all the textfields, labels, and checkboxes

        Panel cboxPanel = new Panel();
        cboxPanel.setLayout(new GridLayout(15,1,2,2));
        cboxPanel.add(panel0);
        cboxPanel.add(panel2);
        cboxPanel.add(panel1);
        cboxPanel.add(panelEq);
        cboxPanel.add(panel3);
        cboxPanel.add(panel5);
        cboxPanel.add(panel4);
        cboxPanel.add(panel6a);
        cboxPanel.add(panel6);
        cboxPanel.add(panel8);
        cboxPanel.add(panel9);
        cboxPanel.add(panel10);
        cboxPanel.add(panel11);
        cboxPanel.add(panel2B);
        cboxPanel.add(panel12);
       // cboxPanel.add(panelA);
       // cboxPanel.add(panel7);
		

        // Add the cboxPanel to Panel p

        this.add(cboxPanel,"West");

        // Add Dismiss, Save Changes, Print, and Help buttons

        Panel botPanel = new Panel();
        botPanel.setFont(buttonFont);
        botPanel.setBackground(MyColors.gray204);
        Label label1 = new Label();
        Label label2 = new Label();
        Label label3 = new Label();
        Label label4 = new Label();

        Button dismissButton = new Button("Cancel");
        Button saveButton = new Button("Save Changes");
        Button printButton = new Button("  Print  ");
        Button helpButton = new Button("  Help  ");
        botPanel.add(label1);
        botPanel.add(dismissButton);
        botPanel.add(label2);
        botPanel.add(saveButton);
        botPanel.add(label3);
        botPanel.add(printButton);
        botPanel.add(label4);
        botPanel.add(helpButton);

        this.add("South", botPanel);


        // Add inner class event handler for Dismiss button

        dismissButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
              hide();
              dispose();
            }
        });


        // Add inner class event handler for Save Changes button

        saveButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){

                try {       // Catch NumberFormatExceptions and warn user

                    // Following String -> double and then cast to int
                    // is to allow exponential notation in the entry
                    // field (e.g., 1.0e5 is valid as an entry now,
                    // but would not be if totalSeeds were converted
                    // directly from String to int.)

                    StochasticElements.Ye = SegreFrame.stringToDouble(Ye.getText().trim());
                    StochasticElements.logtmin = SegreFrame.stringToDouble(logtmin.getText().trim());
                    StochasticElements.logtmax = SegreFrame.stringToDouble(logtmax.getText().trim());
                    StochasticElements.nintervals = SegreFrame.stringToInt(nintervals.getText().trim());

                    // Math.max and Math.min in following to ensure that plotting limits
                    // stored below lie within integration limits stored above
                    
                    StochasticElements.logtmax = Math.max(
                        Cvert.stringToDouble(logtmaxPlot.getText().trim()), StochasticElements.logtmax );

                    StochasticElements.logtminPlot = Math.max(StochasticElements.logtmin,
                        SegreFrame.stringToDouble(logtminPlot.getText().trim()) );
                     StochasticElements.logtmaxPlot = Math.min(StochasticElements.logtmax,
                         SegreFrame.stringToDouble(logtmaxPlot.getText().trim()) );

                    StochasticElements.yminPlot = SegreFrame.stringToDouble(XminPlot.getText().trim());
                    StochasticElements.ymaxPlot = SegreFrame.stringToDouble(XmaxPlot.getText().trim());

                    StochasticElements.xtics = SegreFrame.stringToInt(xtics.getText().trim());
                    StochasticElements.ytics = SegreFrame.stringToInt(ytics.getText().trim());
                    StochasticElements.maxToPlot = SegreFrame.stringToInt(maxCurves.getText().trim());
				
                    StochasticElements.xdeci = SegreFrame.stringToInt(xdeci.getText().trim());
                    StochasticElements.ydeci = SegreFrame.stringToInt(ydeci.getText().trim());

                    StochasticElements.Ymin = SegreFrame.stringToDouble(Ymin.getText().trim());

                    StochasticElements.myComment = commy.getText().trim();
                    
                    StochasticElements.equilibrateTime = SegreFrame.stringToDouble(
                    		eqTime.getText().trim());
                    StochasticElements.equiTol = SegreFrame.stringToDouble(eqTol.getText().trim());

					if(trackEq.getSelectedIndex()==0){
                        StochasticElements.equilibrate=false;
                    } else {
                        StochasticElements.equilibrate=true;
                    }

                    if(energy.getSelectedIndex() == 0) {
                        StochasticElements.plotdE = false;
                        StochasticElements.plotEnergy = true;
                    } else if(energy.getSelectedIndex() == 1){
                        StochasticElements.plotdE = true;
                        StochasticElements.plotEnergy = true;
                    } else {
                        StochasticElements.plotEnergy = false;
                    }

                    if(plotY.getSelectedIndex() == 0) {
                        StochasticElements.plotY =false;
                    } else {
                        StochasticElements.plotY = true;
                    }

                    if(linesOnly.getSelectedIndex() == 0) {
                        StochasticElements.linesOnly =true;
                    } else {
                        StochasticElements.linesOnly = false;
                    }

                    if(blackOnly.getSelectedIndex() == 0) {
                        StochasticElements.blackOnly =false;
                    } else {
                        StochasticElements.blackOnly = true;
                    }

                    if(write3D.getSelectedIndex() == 0) {
                        StochasticElements.write3DOutput =false;
                    } else {
                        StochasticElements.write3DOutput = true;
                    }

                    if(alphaOnly.getSelectedIndex()==0){
                        StochasticElements.tripleAlphaOnly=false;
                    } else {
                        StochasticElements.tripleAlphaOnly=true;
                    }

                    if(renormX.getSelectedIndex() == 0) {
                        StochasticElements.renormalizeMassFractions =false;
                    } else {
                        StochasticElements.renormalizeMassFractions = true;
                    }

                    StochasticElements.minLogContour =
                        SegreFrame.stringToDouble(minContour.getText().trim());

                    if(asymptotic.getSelectedIndex() == 5) {                // F90 asymptotic
                        StochasticElements.integrateWithJava = false;
                    } else if(asymptotic.getSelectedIndex() == 0) {         // Normal asymptotic
                        StochasticElements.doAsymptotic = true;
                        StochasticElements.doSS = false;
                        StochasticElements.asyPC = false;
                        StochasticElements.imposeEquil = false;
                        StochasticElements.integrateWithJava = true;         
                    } else if (asymptotic.getSelectedIndex() == 1) {       // Quasi-steady-state
                        StochasticElements.doSS = true;
                        StochasticElements.doAsymptotic = false;
                        StochasticElements.imposeEquil = false;
                        StochasticElements.integrateWithJava = true;
                    } else if (asymptotic.getSelectedIndex() == 2) {        // Mott asymptotic
                        StochasticElements.doSS = false;
                        StochasticElements.doAsymptotic = true;
                        StochasticElements.asyPC = true;
                        StochasticElements.isMott = true;
                        StochasticElements.imposeEquil = false;
                        StochasticElements.integrateWithJava = true;        
                    } else if (asymptotic.getSelectedIndex() == 3) {       // Oran-Boris asymptotic
                        StochasticElements.doSS = false;
                        StochasticElements.doAsymptotic = true;
                        StochasticElements.asyPC = true;
                        StochasticElements.isMott = false;
                        StochasticElements.imposeEquil = false;
                        StochasticElements.integrateWithJava = true;
                    } else if (asymptotic.getSelectedIndex() == 6) {       // Asy + PE
                    	StochasticElements.doAsymptotic = true;
                        StochasticElements.doSS = false;
                        StochasticElements.asyPC = false;
                        StochasticElements.imposeEquil = true;
                        StochasticElements.integrateWithJava = true;
                    } else if (asymptotic.getSelectedIndex() == 7) {       // QSS + PE
                    	StochasticElements.doAsymptotic = false;
                        StochasticElements.doSS = true;
                        StochasticElements.asyPC = false;
                        StochasticElements.imposeEquil = true;
                        StochasticElements.integrateWithJava = true;
                    } else if (asymptotic.getSelectedIndex() == 4) {          // Explicit
                        StochasticElements.doSS = false;
                        StochasticElements.doAsymptotic = false;
                        StochasticElements.imposeEquil = false;
                        StochasticElements.integrateWithJava = true;
                    } else {
                    	Cvert.callExit("*** Call exit: inconsistent integration method choice");
                    }

                    StochasticElements.massTol = SegreFrame.stringToDouble(massTol.getText().trim());

                    if(lpFormat.getSelectedIndex() == 0) {
                        StochasticElements.longFormat = false;
                    } else {
                        StochasticElements.longFormat = true;
                    }

                    StochasticElements.popColorMap = popCM.getSelectedItem();
                    
                    StochasticElements.fluxColorMap = fluxCM.getSelectedItem();
                    
                    if( checkBox[1].getState() ) {
                        StochasticElements.rho = SegreFrame.stringToDouble(rho.getText().trim());
                        StochasticElements.T9 = SegreFrame.stringToDouble(T9.getText().trim());
                        StochasticElements.constantHydro = true;
                    } else {
                        StochasticElements.profileFileName = profile.getText();
                        StochasticElements.constantHydro = false;
                    }
                    StochasticElements.stochasticFactor = SegreFrame.stringToDouble(sf.getText().trim());

// Remove following for now
/*
                    if( (byte)SegreFrame.stringToInt(zmax.getText()) > StochasticElements.pmax 
                                || (byte)SegreFrame.stringToInt(nmax.getText())>StochasticElements.nmax ){
                            String message = "Zmax can't be greater than ";
                            message += StochasticElements.pmax;
                            message += " (set by pmax in StochasticElements)";
                            message+= " and Nmax can't be greater than ";
                            message += StochasticElements.nmax;
                            message += " (set by nmax in the class StochasticElements).";
                            message += " Change Zmax and/or Nmax entries to conform, or";
                            message += " change pmax or nmax in StochasticElements.";
                            message += " There must also exist files in the date subdirectory";
                            message += " isoZ_N.ser corresponding to the ranges of Z and N chosen.";
                            makeTheWarning(300,300,250,250,Color.black,
                                    Color.yellow, " Error!", message,
                                    false, ParamSetup.this);
                            return;
                    } else {
                            StochasticElements.pmax = (byte)
                                    SegreFrame.stringToInt(zmax.getText().trim());
                            StochasticElements.nmax = (byte)
                                    SegreFrame.stringToInt(nmax.getText().trim());
                            StochasticElements.pmin = (byte)
                                    SegreFrame.stringToInt(zmin.getText().trim());
                    }

*/

                    hide();
                    dispose();

                } catch(NumberFormatException e) {
                    String screwup = "NumberFormatException.";
                    screwup += "  At least one required data field has";
                    screwup += " a blank or invalid entry.";
                    System.out.println(screwup);
                    makeTheWarning(300,300,200,150,Color.black,
                        Color.lightGray, " Warning!",
                        screwup, false , ParamSetup.this);
                }
            }
        });  // -- end inner class for Save button processing


        // Help button actions.  Handle with an inner class.
        
        helpButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
        
                if(helpWindowOpen) {
                    hf.toFront();
                } else {
                    hf = new GenericHelpFrame(ParamSetup.makeHelpString(),
                        " Help for parameter setup",
                        500,400,10,10,200,10);
                    hf.show();
                    helpWindowOpen = true;
                }
            }
        });


        // Add inner class event handler for Print button

        printButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
                printThisFrame(20,20,false);
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



    // -------------------------------------------------------------------------------------------
    //  Method to act when state of Checkboxes changes.
    //  Requires that the class implement the
    //  ItemListener interface, which in turn requires that the
    //  method itemStateChanged be defined explicitly since
    //  ItemListener is abstract.
    // --------------------------------------------------------------------------------------------

    public void itemStateChanged(ItemEvent check) {

        // Process the reaction class checkboxes.  First
        // get the components of the relevant panels
        // and store in Component arrays (Note: the method
        // getComponents() is inherited from the Container
        // class by the subclass Panel).

        Component [] components4 = panel4.getComponents();
        Component [] components5 = panel5.getComponents();

        // Now process these components that are checkboxes
        // (only the first element of each array is).  First cast the
        // Component to a Checkbox.  Then use the getState()
        // method of Checkbox to return boolean true if
        // checked and false otherwise.

        Checkbox cb4 = (Checkbox)components4[0];  // Checkbox for panel4
        Checkbox cb5 = (Checkbox)components5[0];  // Checkbox for panel5

        // Then use the getState() method of Checkbox to
        // return boolean true if checked and false otherwise.
        // Use this logic to disable one or the other sets of
        // choices for temperature and density input.

        if( cb4.getState() ) {
           checkBox[1].setState(false); // Seems needed despite CheckBoxGroup
           rho.disable();
           rho.setBackground(disablebgColor);
           rhoL.setForeground(disablefgColor);
           T9.disable();
           T9.setBackground(disablebgColor);
           T9L.setForeground(disablefgColor);
           profile.enable();
           profile.setBackground(panelBackColor);
           profileL.setForeground(panelForeColor);
        } else if ( cb5.getState() ) {
           checkBox[0].setState(false);
           rho.enable();
           rho.setBackground(panelBackColor);
           rhoL.setForeground(panelForeColor);
           T9.enable();
           T9.setBackground(panelBackColor);
           T9L.setForeground(panelForeColor);
           profile.disable();
           profile.setBackground(disablebgColor);
           profileL.setForeground(disablefgColor);
       }
    }


    // ----------------------------------------------------------------------------------------------
    //  Method printThisFrame prints an entire Frame (Java 1.1 API).
    //  To print a Frame or class subclassed from Frame, place this
    //  method in its class description and invoke it to print.
    //  It MUST be in a Frame or subclass of Frame.
    //  It invokes a normal print dialogue,
    //  which permits standard printer setup choices. I have found
    //  that rescaling of the output in the print dialogue works
    //  properly on some printers but not on others.  Also, I tried
    //  printing to a PDF file through PDFwriter but that caused
    //  the program to freeze, requiring a hard reboot to recover.
    //  (You can use the PSfile method to output to a .ps file.)
    //
    //      Variables:
    //      xoff = horizontal offset from upper left corner
    //      yoff = vertical offset from upper left corner
    //      makeBorder = turn rectangular border on or off
    // ------------------------------------------------------------------------------------------------

    public void printThisFrame(int xoff, int yoff, boolean makeBorder) {

        java.util.Properties printprefs = new java.util.Properties();
        Toolkit toolkit = this.getToolkit();
        PrintJob job = toolkit.getPrintJob(this,"Java Print",printprefs);
        if (job == null) {return;}
        Graphics g = job.getGraphics();
        g.translate(xoff,yoff);     // Offset from upper left corner
        Dimension size = this.getSize();
        if (makeBorder) {           // Rectangular border
            g.drawRect(-1,-1,size.width+2,size.height+2);
        }
        g.setClip(0,0,size.width,size.height);
        this.printAll(g);
        g.dispose();
        job.end();
    }




    // --------------------------------------------------------------------------------------------------------------------
    //  The method makeTheWarning creates a modal warning window when invoked
    //  from within an object that subclasses Frame. The window is
    //  modally blocked (the window from which the warning window is
    //  launched is blocked from further input until the warning window
    //  is dismissed by the user).  Method arguments:
    //
    //      X = x-position of window on screen (relative upper left)
    //      Y = y-position of window on screen (relative upper left)
    //      width = width of window
    //      height = height of window
    //      fg = foreground (font) color
    //      bg = background color
    //      title = title string
    //      text = warning string text
    //      oneLine = display as one-line Label (true)
    //                or multiline TextArea (false)
    //      frame = A Frame that is the parent window modally
    //              blocked by the warning window.  If the parent
    //              class from which this method is invoked extends
    //              Frame, this argument can be just "this" (or
    //              "ParentClass.this" if invoked from an
    //              inner class event handler of ParentClass).
    //              Otherwise, it must be the name of an object derived from
    //              Frame that represents the window modally blocked.
    //
    //  If oneLine is true, you must make width large enough to display all
    //  text on one line.
    // ---------------------------------------------------------------------------------------------------------------------

    public void makeTheWarning (int X, int Y, int width, int height,
        Color fg, Color bg, String title, String text, boolean oneLine, Frame frame) {

        Font warnFont = new java.awt.Font("SanSerif", Font.BOLD, 12);
        FontMetrics warnFontMetrics = getFontMetrics(warnFont);

        // Create Dialog window with modal blocking set to true.
        // Make final so inner class below can access it.

        final Dialog mww = new Dialog(frame, title, true);
        mww.setLayout(new BorderLayout());
        mww.setSize(width,height);
        mww.setLocation(X,Y);

        // Use Label for 1-line warning

        if (oneLine) {
            Label hT = new Label(text,Label.CENTER);
            hT.setForeground(fg);
            hT.setBackground(bg);
            hT.setFont(warnFont);
            mww.add("Center", hT);

        // Use TextArea for multiline warning

        } else {
            TextArea hT = new TextArea("",height,width, TextArea.SCROLLBARS_NONE);
            hT.setEditable(false);
            hT.setForeground(fg);
            hT.setBackground(bg);  // no effect once setEditable (false)?
            hT.setFont(warnFont);
            mww.add("Center", hT);
            hT.appendText(text);
        }

        mww.setTitle(title);

        // Add dismiss button

        Panel botPanel = new Panel();
        botPanel.setBackground(Color.lightGray);
        Label label1 = new Label();
        Label label2 = new Label();

        Button dismissButton = new Button("Dismiss");
        botPanel.add(label1);
        botPanel.add(dismissButton);
        botPanel.add(label2);

        // Add inner class event handler for Dismiss button.  This must be
        // added to the dismissButton before botPanel is added to mww.

        dismissButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent ae){
              mww.hide();
              mww.dispose();
            }
        });

        mww.add("South", botPanel);

        // Add window closing button (inner class)

        mww.addWindowListener(new WindowAdapter() {
           public void windowClosing(WindowEvent e) {
              mww.hide();
              mww.dispose();
           }
        });

        mww.show();             // Note that this show must come after all the above
                                // additions; otherwise they are not added before the
                                // window is displayed.
    }




    // ----------------------------------------------------------------------
    //  Static method to generate string for Help file
    // ----------------------------------------------------------------------

    static String makeHelpString() {

        String s;
        s="The parameters set through this interface control many of the";
        s+=" details of the calculation.  Once the parameters are set,";
        s+=" store them by clicking the \"Save Changes\" button.\n\n";

        s+="Log10 Start Time\nBase-10 logarithm of the start time for";
        s+=" integration, in seconds. Distinguish from logx- below, which is";
        s+= " the start time for the plot. \n\n";

        s+="Log10 END TIME\nBase-10 logarithm of the final time for";
        s+=" integration, in seconds. Distinguish from logx+ below, which is";
        s+= " the end time for the plot. \n\n";

        s+="Precision\nPrecision factor for integration timestep.";
        s+=" Smaller values give smaller adaptive timestep and more precision.\n\n";

        s+="CalcMode\nCalculation mode (QSS=quasi-steady-state, Asy=normal asymptotic,";
        s+=" AsyMott=Mott asymptotic, AsyOB=Oran-Boris asymptotic, Explicit=explicit, F90Asy=F90).\n\n";

        s+="dX\nPermitted tolerance in deviation of sum of mass fractions X";
        s+=" from unity. For example, a value of 0.01 requires the timestep to be";
        s+=" adjusted such that sum X lies between 0.99 and 1.01";
        s+=" after each timestep.\n\n";

//         s+="Max Light-Ion Mass\nMaximum mass light ion permitted to";
//         s+=" contribute to reactions.  For example, if this variable is set";
//         s+=" to 4, p + O16 and He4 + O16 reactions can contribute (if";
//         s+=" the corresponding Reaction Class is selected), but C12 + O16";
//         s+=" would be excluded since 12 is larger than 4.  As another";
//         s+=" example, if Reaction Class 5 is selected and this variable";
//         s+=" is set equal to 1, p-gamma reactions contribute but not";
//         s+=" alpha-p reactions.\n\n";

        s+="HYDRODYNAMIC VARIABLES\nThe temperature (in units";
        s+=" of 10^9 K), the density (in units of g/cm^3), and electron fraction Ye are set in";
        s+=" one of two mutually exclusive ways, chosen through the radio";
        s+=" buttons.  If \"Constant\" is selected the (constant)";
        s+=" temperature (T), density (rho), and electron fraction (Ye) are entered in the";
        s+=" corresponding fields.  If \"Specify Profile\"";
        s+=" is selected instead, a file name is specified that";
        s+=" contains the time profile for T, rho, and Ye to be used in the";
        s+=" calculation";
        s+=" (note: assume to be case-sensitive) if in the same directory";
        s+=" as the Java class files, or by a properly qualified path";
        s+=" for the machine in question if in another directory.  For";
        s+=" example, on a Unix machine \"../file.ext\" would specify the";
        s+=" file with name \"file.ext\" in the parent directory.";
        s+=" See the file jin/sampleHydroProfile.inp for sample file format\n\n";


        s+="logx-\nThe lower limit in log time for plot.";
        s+=" This differs from LOG10 START TIME, which is the integration start time.";
        s+=" Generally logx- must be greater than or equal to LOG10 START TIME ";
        s+= " (the program will set logx- = LOG10 START TIME otherwise).\n\n";

        s+="logx+\nThe upper limit in log time for plot.";
        s+=" This differs from LOG10 END TIME, which is the integration stop time.";
        s+=" Generally logx+ must be less than or equal to LOG10 END TIME ";
        s+= " (the program will set logx+ = LOG10 END TIME otherwise).\n\n";

        s+="logy-\nThe lower limit in log X (mass fraction) or log Y (abundance) for plot";
        s+= " (which one is determined by the X/Y selection widget).\n\n";

        s+="logy+\nThe upper limit in log X (mass fraction) or log Y (abundance) for plot.";
        s+= " (which one is determined by the X/Y selection widget).\n\n";

        s+="x tics\nNumber of tic marks on x axis.\n\n";

        s+="y tics\nNumber of tic marks on y axis.\n\n";

        s+="Isotopes\nMaximum number of curves to plot (500 is the largest permitted value).\n\n";

        s+="E/dE\nWhether integrated energy E or differential energy dE is plotted.\n\n";

        s+="X/Y\nWhether mass fraction X or molar abundance Y is plotted.\n\n";

        s+="Lines/Symbols\nWhether curves are drawn with lines or series of symbols.\n\n";

        s+="Color/BW\nWhether curves are drawn in color or black and white.\n\n";

        s+="Steps\nNumber of time intervals to plot (time intervals";
        s+=" will be equally spaced on log scale). Typical values are ~100. Max is";
        s+=" StochasticElements.tintMax. Larger values give better plot resolution but"; 
        s+=" larger postscript files.\n\n";

        s+="MinCon\nApproximate minimum contour for 2D and 3D plots.\n\n";

        s+="3D\nWhether to output data in ascii file 3D.data for 3D plot animation by rateViewer3D.\n\n";

        s+="AlphaOnly\nIf No, include only triple-alpha among alpha reactions with light ions";
        s+=" (for alpha network).  Otherwise, include all alpha reactions with light ions.\n\n";

        s+="x deci\nNumber of digits beyond decimal for x-axis labels.\n\n";

        s+="y deci\nNumber of digits beyond decimal for y-axis labels.\n\n";

        s+="Ymin\nFractional particle number threshold for including a box in network in a given";
        s+=" timestep. Roughly, in given timestep only isotopes having an an abundance Y";
        s+=" larger than Ymin will be processed.\n\n";

        s+="renormX\nWhether to renormalize sum of mass fractions to 1 after each timestep.\n\n";
        
        s+="Lineplot\nDetermines whether the lineplot produced";
        s+=" of populations as a function of time has a tall or ";
        s+=" short vertical aspect.\n\n";
        
        s+="PopCM\nChooses colormap for 2D population animation.\n\n";
        
        s+="FluxCM\nChooses colormap for 2D flux ratio animation.\n\n";

        s+="Comments\nOptional comments that will be included in output text and graphics.\n\n";

// Remove following for now
/*
        s+="Zmax\nThe maximum value of proton number that will be considered";
        s+=" in the calculation.  Set larger than the highest Z likely";
        s+=" to be encountered in the reactions of the network.";
		s+=" Can't be larger than StochasticElements.pmax.\n\n";

        s+="Nmax\nThe maximum value of neutron number that will be considered";
        s+=" in the calculation.  Set larger than the highest N likely";
        s+=" to be encountered in the reactions of the network.";
		s+=" Can't be larger than StochasticElements.nmax.\n\n";

        s+="Zmin\nThe minimum value of proton number that will be considered";
        s+=" in the calculation for heavier ions. (Note: protons, neutrons,";
        s+=" He3, and He4 are always considered, irrespective of this";
        s+=" parameter).  For example, if Zmin=6, no isotopes with mass";
        s+=" numbers less than 6 will be considered, except for protons,";
        s+=" neutrons (if Include Neutrons is set to \"Yes\"), He3, and";
        s+=" alpha particles. Normally changed only for diagnostic purposes. \n\n";
*/

        return s;

    }

}   /* End class ParamSetup */