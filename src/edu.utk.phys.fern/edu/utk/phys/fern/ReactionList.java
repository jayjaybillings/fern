package edu.utk.phys.fern;
// -------------------------------------------------------------------------------------------
//  Class ReactionList creates a scrollable panel of reactions with checkboxes to
//  permit the reactions to be selected.
// -------------------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import gov.sandia.postscript.PSGr1;


class ReactionList extends Panel {

    Font titleFont = new java.awt.Font("SanSerif", Font.BOLD, 12);
    FontMetrics titleFontMetrics = getFontMetrics(titleFont);
    Font buttonFont = new java.awt.Font("SanSerif", Font.BOLD, 11);
    FontMetrics buttonFontMetrics = getFontMetrics(buttonFont);

    Color panelForeColor = Color.black;
    Color panelBackColor = new Color(240,240,240);
    Color color1 = new Color(215,215,215);
    Color color2 = new Color(230,230,230);

    int nR, len;       // Number of reactions
    Checkbox [] cb = new Checkbox[50];  // Max of 50 entries

    ReactionClass1 [] rArray = new ReactionClass1[50];

    int Z,N;

    // serialNumber holds the serialIndex that uniquely identifies a reaction
    // by giving its sequence number in the deserialization from the disk file.

    int [] serialNumber = new int [50];


    // ----------------------------------------------------------------------------
    //  Constructor
    // ----------------------------------------------------------------------------

    public ReactionList (){

        Z = IsotopePad.protonNumber;
        N = IsotopePad.neutronNumber;

        loadData();

        //nR = len;

        this.setLayout(new GridLayout(nR,1,5,5));

        //  Create a total of len checkboxes, but only display them
        //  initially if the corresponding reaction class is selected

        boolean setColor1 = false;

        for(int i=0; i<nR; i++) {
            String cbString = " "+rArray[i].reacString+"  Q = "
                +rArray[i].Q+"  Class = " +rArray[i].reacIndex
                +"  Serial = "+serialNumber[i];
            cb[i] = new Checkbox(cbString,false);
            cb[i].setBackground(color2);
			// Reaction class if class selected
            int rc = rArray[i].reacIndex;  
            if(IsoData.checkBox[rc].getState()) {  
                this.add(cb[i]);
                if(!DataHolder.RnotActive[Z][N][i]) {
                    DataHolder.RnotActive[Z][N][i] = 
						!StochasticElements.pruneReactions(Z, N, rArray[i]);
                }
                cb[i].setState(!DataHolder.RnotActive[Z][N][i]);
            }
        }
    }



    // ----------------------------------------------------------------------------
    //  Method to load data into reaction array rArray[] by
    //  deserializing reaction objects from disk
    // ----------------------------------------------------------------------------

    void loadData() {

        // Construct name of the serialized file corresponding
        // to this isotope.  These files are produced by the
        // class FriedelParser from the Thielemann reaction
        // library.  They should be in a subdirectory of the present
        // directory called "data", and their names should have
        // the standard form "isoZ_N.ser", where Z is the
        // proton number and N the neutron number of the isotope.

        String file = "data/iso" + Z + "_" + N + ".ser";

        try {

            // Wrap input file stream in an object input stream

            FileInputStream fileIn = new FileInputStream(file);
            ObjectInputStream in = new ObjectInputStream(fileIn);

            // Read from the input stream the initial integer giving
            // the number of objects that were serialized in this file

            int numberObjects = in.readInt();

            // Read from the input stream the 9-member int array giving
            // the number of reactions of each type.  Entry 0 is
            // the total (=numberObjects).  Array entries 1-8 give the
            // the subtotals for each of the 8 reaction types

            int [] numberEachType = (int []) in.readObject();

            // Calculate the number of objects to read in

            int [] tempArray = new int[9];
            tempArray[0] = 0;
            for (int q=1; q<9; q++) {
                tempArray[q] = 1;
            }

            // Deserialize the objects to the array rArray []

            len = numberObjects;
            int m=0;
            int mm=0;
            while (mm < len) {
                ReactionClass1 tryIt = (ReactionClass1)in.readObject();
                boolean selectFlag = true;

                // Restrict display for He-4 to light-ion with light-ion reactions
                // and likewise with p or n reactions, since there are too many
                // of them and the ones that are not light ion on light ion are
                // already included under the heavy target isotope.
                
                if(Z==2 && N==2){
                    selectFlag=false;
                    if(StochasticElements.isLightIonReaction(tryIt)){
                        selectFlag = true;
                    }
                } else if (( Z==1 && N==0) || (Z==0 && N==1)) {
                    selectFlag=false;
                    if(StochasticElements.isLightIonReaction(tryIt)){
                        selectFlag = true;
                    }
                }

                if( tempArray[tryIt.reacIndex] == 1 && selectFlag ) {
                    rArray[m] = tryIt;
                    serialNumber[m] = mm;
                    m++;
                }

                if(tempArray[tryIt.reacIndex] == 1) { mm ++; }

            }

            nR = m;

            // Close the input streams

            in.close();
            fileIn.close();
        }                                        // -- end try
        catch (Exception e) {
            System.out.println(e);
        }

    }  /* End method loadData */

}   /* End class ReactionList */

