// --------------------------------------------------------------------------------------------------------------------------------
//  Class to read in objects originally serialized using the class ReactionClass1 and output
//  the reaction information to an ascii file 'outfile' in a format specified by the method
//  outputReactionData.  Invoke from the command line:
//
//      java DeserializeReaclib minZ maxZ minN maxN outputFile
//
//  where minZ and maxZ specify the range of proton numbers and minN and maxN the
//  range of neutron numbers to consider, and outputFile is the name of the output ascii file
//  of reaction information.  Requires the class ReactionClass1.
// 
// --------------------------------------------------------------------------------------------------------------------------------


import java.io.*;

class DeserializeReaclib {
    
    static int minZ;
    static int maxZ;
    static int minN;
    static int maxN;
    static int reactionClass;
    String inputFile;
    static String outputFile;
    static FileOutputStream outstream;
    static PrintWriter toOut;
    boolean orderByReactionClass = true;
    static int numberObjects;
    static int numberMatches;
    

    public static void main(String[] args) {  
    
        // Check for correct number of command-line arguments

        if (args.length != 5) {
            System.err.println();
            System.err.println("Usage: java DeserializeReaclib Zmin Zmax Nmin Nmax outputFile");
            System.err.println();
            System.exit(1);
        }
        
        minZ = Integer.parseInt(args[0]);
        maxZ = Integer.parseInt(args[1]);
        minN = Integer.parseInt(args[2]);
        maxN = Integer.parseInt(args[3]);
        outputFile = args[4];
        
        try{       
            outstream = new FileOutputStream(outputFile);
            toOut = new PrintWriter(outstream);
        
            // Loop over isotopes and deserialize
            
            // Order according to reaction classes 1-8
            for(int c = 1; c<9; c++){
                reactionClass = c;
                System.out.println("\nReaction Class = "+reactionClass);
                toOut.println(c);
                toOut.println(); toOut.println();
                // Loop over proton number and neutron number
                for(int Z=minZ; Z<=maxZ; Z++){
                    for(int N=minN; N<=maxN; N++){
                        deserializeFile(Z,N);           
                    }
                }  
            }        
        }
        catch (IOException e) { ; }
        
        toOut.flush();
        toOut.close();
        
        try{outstream.close();} 
        catch (IOException e) {;}         
    }
    
    
    
    // ------------------------------------------------------------------------------------------------------------------------
    //  Method to deserialize file containing the reaction information for the isotope Z, N
    // ------------------------------------------------------------------------------------------------------------------------
    
    public static void deserializeFile(int Z, int N){
        
        // Return if the file does not exist
        
        String inputFile = "data/iso"+Z+"_"+N+".ser";
        File test = new File(inputFile);
        if(!test.exists()){
            //System.out.println("No file "+inputFile);
            return;
        }

        // If the file exists, create an input object stream wrapped around a file input
        // stream.  Then read the initial Integer stored giving the
        // number of objects that were serialized.  Then
        // read the objects in sequentially, first
        // casting to the correct type (the return type of readObject()
        // is Object, so we have to cast this to the ReactionClass1
        // type that we expect) and then assigning to elements
        // of the ReactionClass1 array instance[].

        try {

            // Take the name of the input file from standard input;
            // wrap this file in an object input stream        

            FileInputStream fileIn = new FileInputStream(inputFile);
            ObjectInputStream in = new ObjectInputStream(fileIn);

            // Read from the input stream the initial integer giving
            // the number of objects that were serialized in this file

            numberObjects = in.readInt();
//             System.out.println();
            System.out.println("Z="+Z+" N="+N+" numberObjects="+numberObjects);
//             System.out.println();

            // Read from the input stream the 9-d int array giving
            // the number of reactions of each type.  Entry 0 is
            // the total (numberObjects).  Array entries 1-8 give the
            // the subtotals for each of the 8 reaction types

            int [] numberEachType = (int []) in.readObject();
//             System.out.println("Number of Reactions of All Types = "
//                 + numberEachType[0]);
//             for (int k=1; k<9; k++) {
//                 System.out.println("Number Reactions of Type "
//                      + k + " = " + numberEachType[k]);
//             }
//             System.out.println();

            // Create an array to hold the reaction objects as they are
            // deserialized

            ReactionClass1 [] instance = new ReactionClass1 [numberObjects];

            // Deserialize the objects to the array instance[]

            int i=0;
            int k=0;
            while (i < numberObjects) {
                ReactionClass1 temp = (ReactionClass1)in.readObject();
                // Require match to current reaction class        
                if(temp.reacIndex == reactionClass){
                    instance[k] = temp;
                    k++;
                }
                i++;
            }
            
            numberMatches = k;     // Number of reactions for this Z and N matching current reaction class

            // Close the input streams

            in.close();
            fileIn.close();
            
            // Output the reaction information for this Z, N
            
            outputReactionData(instance);

        }
        catch (Exception e) {
            System.out.println(e);
        }
    }
    
   
    // -------------------------------------------------------------------------------------------------------------------   
    //  Method to output the deserialized reaction information to an ascii output file.
    //  This can be customized to give the desired output file format.
    // -------------------------------------------------------------------------------------------------------------------
    
    public static void outputReactionData(ReactionClass1 [] instance){
        
        // Print out the instance variables from the serialized objects that we have just read
        
        String spacer = " ";
        String symb = "";
        String ts = "";
        
        for (int j=0; j < numberMatches; j++) {
            
            ts = "   ";
            
            for(int k=0; k<instance[j].numberReactants; k++){
                int z = instance[j].isoIn[k].x;
                int n = instance[j].isoIn[k].y;
                int massNumber = n+z;
                if(z==1){
                    switch(n){
                        case 0:
                            symb = "p";
                            break;
                        case 1:
                            symb = "d";
                            break;
                        case 2:
                            symb = "t";
                            break;
                    }
                } else {
                    symb = Cvert.returnSymbol(z).toLowerCase();
                }
                if(z > 1) symb = symb+massNumber;
                ts += (symb+spacer);            
            }
            
            for(int k=0; k<instance[j].numberProducts; k++){
                int z = instance[j].isoOut[k].x;
                int n = instance[j].isoOut[k].y;
                int massNumber = n+z;
                if(z==1){
                    switch(n){
                        case 0:
                            symb = "p";
                            break;
                        case 1:
                            symb = "d";
                            break;
                        case 2:
                            symb = "t";
                            break;
                    }
                } else {
                    symb = Cvert.returnSymbol(z).toLowerCase();
                }
                if(z > 1) symb = symb+massNumber;
                ts += (symb + spacer);            
            }
            
            symb = instance[j].refString;
            if(instance[j].resonant) {
                symb += "r";
            } else if (instance[j].nonResonant){
                symb += "n";
            }
                
            if(instance[j].reverseR) symb += "v";
            
            ts += (symb+spacer);
            
            ts += (instance[j].Q);
            
            ts += ("                 Diagnostic: "+instance[j].reacString+" class="+instance[j].reacIndex);
        
            toOut.println(ts);
            
            ts = ""+instance[j].p0+spacer+instance[j].p1+spacer+instance[j].p2+spacer+instance[j].p3;      
            toOut.println(ts);     
            ts = ""+instance[j].p4+spacer+instance[j].p5+spacer+instance[j].p6;      
            toOut.println(ts);
            
        }
        
    }

}  /* End class */