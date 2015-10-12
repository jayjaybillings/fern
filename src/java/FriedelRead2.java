// -------------------------------------------------------------------------------------------------------
// Class to read the Thielemann reaction library, parse the input data,
// and output to another file in a more reader-friendly format with the
// Z and N of all reactants identified. This class can be modified
// easily to output the library in an arbitrary format.  Usage:
//
//          java FriedelRead2 inputFile outputFile
//
// where inputFile contains the Thielemann reaction library in standard
// format (my current version is called netsu) and outputFile is the
// desired name of the output file.
//
//                        --- Mike Guidry (guidry@utk.edu) Dec. 28, 2001)
// ------------------------------------------------------------------------------------------------------


import java.io.*;
import java.lang.*;
import java.util.*;

public class FriedelRead2 {

    static FileOutputStream to;
    static PrintWriter toChar;
    static String s;
    static int buffNumber;

    public static void main(String[] args) {

        // Read in input and output filenames

        if (args.length != 2) {      // Check for fileName arguments
            System.err.println();
            System.err.println("Usage: java FriedelRead2 <inFile> <outFile>");
            System.err.println();
            System.exit(1);
        }
        else {
            // Call copy() to do the copy; display any error messages
            try { copy(args[0], args[1]); }
            catch (IOException e) { System.err.println(e.getMessage()); }
        }

        // Parse the input data and output it

        parseBuffer();

        // Flush and close the i/o streams

        // No exceptions to be caught for following because methods
        // of PrintWriter never throw exceptions.  Must flush toChar
        // before toChar and to are closed to ensure all data gets
        // output.

        toChar.flush();
        toChar.close();

        // But exceptions must be caught for the following

        try {
            to.close();
        } catch (IOException e) {;}

    }


    // ---------------------------------------------------------------------
    //  Method parseBuffer to parse the input buffer
    // ---------------------------------------------------------------------

    static void parseBuffer() {

        // Array holding reaction type & number reactants for that type.
        // For convenience, ignore 0 entry so we can number from 1.

        int [] reactantNumber = new int[9];
        reactantNumber[0]=0;   // Not used
        reactantNumber[1]=1;
        reactantNumber[2]=1;
        reactantNumber[3]=1;
        reactantNumber[4]=2;
        reactantNumber[5]=2;
        reactantNumber[6]=2;
        reactantNumber[7]=2;
        reactantNumber[8]=3;

        // Array holding reaction type descriptive labels.
        // For convenience, ignore 0 entry so we can number from 1.

        String [] reactionType = new String [9];
        reactionType[0] = "";      // Not used
        reactionType[1] = "Decays";
        reactionType[2] = "a -> b + c";
        reactionType[3] = "a -> b + c + d";
        reactionType[4] = "a + b -> c";
        reactionType[5] = "a + b -> c + d";
        reactionType[6] = "a + b -> c + d + e";
        reactionType[7] = "a + b -> c + d + e + f";
        reactionType[8] = "a + b + c -> d (+ e)";

        int numberLeft = 1;      //  Number reactants (left side of equation)
        double Qvalue;           //  Q for reaction
        int reacIndex = 1;

        String [] tempTokens = new String [10];
        String [] reactionComponents = new String[6];   // Components of e.g. n + p -> d
        double [] params = new double[7];                      // 7 parameters for rates

        // Break string into tokens with whitespace delimiter

        StringTokenizer st = new StringTokenizer(s);
        int numberTokens = st.countTokens();
        System.out.println("numberTokens = "+numberTokens);
        int tokenIndex = 0;      // Number Tokens within this buffer
        buffNumber ++;           // Count buffers we've filled

        // Process the string tokens

        st.nextToken();                // Skip the first token
        stopProcessing:                // This is break label
        while(st.hasMoreTokens()){
            int loopCounter = 0;

            // Find the Q-value entry (marks end of first line
            // in a reaction entry).

            for (int i = 0; i<=9; i++) {
                tokenIndex ++;
                loopCounter ++;
                if (!st.hasMoreTokens()) {break stopProcessing;}
                tempTokens[i] = st.nextToken();
                int testE = tempTokens[i].indexOf("E");
                if (testE != -1) {break;}   // Found Q-value entry
            }

            // Each of the 8 reaction types in Friedel library has
            // a header set of lines like a reaction entry, but
            // with only two entries on the first line:  the
            // reaction type as an integer (1-8) and an entry in
            // the Q-value position of 0.00000E00.  The
            // parameters on the next line are also all zero.
            // Detect and skip such headers (except set the
            // reaction index) by noting that the first line has
            // only two tokens, while a legitimate reaction has
            // at least 5.

            if (loopCounter <= 2) {   // Header lines case: skip
                reacIndex = Integer.parseInt(tempTokens[0]);
                numberLeft = reactantNumber[reacIndex];
                for (int i=0; i<=6; i++) {
                    st.nextToken();   // Skip all tokens in header
                    tokenIndex ++;
                }
            } else {                  // But if a real entry

                // First, construct reaction string

                for (int i=1; i<=loopCounter-3; i++) {
                    reactionComponents[i-1] = tempTokens[i];
                }
                String leftSide = "";
                String rightSide = "";

                // Process left side of reaction

                for (int i=0; i<=numberLeft-1; i++){
                    leftSide += reactionComponents[i];
                    if(i<=numberLeft-2) {leftSide += " + ";}

                    // Process reaction symbols according
                    // to byte content

                    byte[]reacBytes=reactionComponents[i].getBytes();
                    int [] tempZN = parseNucSymbols(reacBytes);
                    int Z = tempZN[0];
                    int N = tempZN[1];
                    int mass = Z+N;
                }

                // Process right side of reaction

                for (int i=numberLeft; i<=loopCounter-4; i++) {
                    rightSide += reactionComponents[i];
                    if(i<=loopCounter-5) {rightSide += " + ";}

                    // Process reaction symbols according
                    // to byte content

                    byte[]reacBytes=reactionComponents[i].getBytes();
                    int [] tempZN = parseNucSymbols(reacBytes);
                    int Z = tempZN[0];
                    int N = tempZN[1];
                    int mass = Z+N;
                 }

                 // Add additional species to the
                 // reaction string for weak interactions
                 // (Friedel tables show only nuclear species, not
                 // electrons or neutrinos)

                 boolean ecFlag = false;   // true if electron capture

                 if(tempTokens[loopCounter-2].compareTo("bet-") == 0) {
                     rightSide += " + e + nubar";
                 }
                 if(tempTokens[loopCounter-2].compareTo("bet+") == 0) {
                     rightSide += " + e+ + nu";
                 }
                 if(tempTokens[loopCounter-2].compareTo("btyk") == 0) {
                     rightSide += " + e+ + nu";
                 }
                 if(tempTokens[loopCounter-2].compareTo("ec") == 0) {
                     leftSide += " + e";
                     rightSide += " + nu  (electron capture)";
                     ecFlag = true;
                 }
                 if(tempTokens[loopCounter-2].compareTo("bec") == 0) {
                     leftSide += " + e";
                     rightSide += " + nu  (electron capture)";
                     ecFlag = true;
                 }
                 leftSide += " --> ";

                 // Save the reaction reference string and parse it
                 // for more information about the reaction

                 String refString = tempTokens[loopCounter-2];
                 boolean reverseR = false;         // true indicates reverse reac
                 boolean resonant = false;         // true indicates resonant
                 boolean nonResonant = false;  // true indicates nonresonant
                 int refL = refString.length();

                 // Check the last character

                 String check = refString.substring(refL-1,refL);
                 if (check.compareTo("v") == 0) {reverseR = true;}
                 if (check.compareTo("n") == 0) {nonResonant = true;}
                 if (check.compareTo("r") == 0) {resonant = true;}

                 // Check the next-to-last character

                 check = refString.substring(refL-2,refL-1);
                 if (check.compareTo("n") == 0) {nonResonant = true;}
                 if (check.compareTo("r") == 0) {resonant = true;}

                 String rDetails = " ";
                 if(reverseR && !resonant && !nonResonant) {
                     rDetails = " [reverse] ";
                 }
                 if(reverseR && resonant && !nonResonant) {
                     rDetails = " [resonant,reverse] ";
                 }
                 if(reverseR && !resonant && nonResonant) {
                     rDetails = " [nonresonant,reverse] ";
                 }
                 if(!reverseR && resonant && !nonResonant) {
                     rDetails = " [resonant] ";
                 }
                 if(!reverseR && !resonant && nonResonant) {
                     rDetails = " [nonresonant] ";
                 }

                 // Extract Q-value as a double from token

                 Qvalue = stringToDouble(tempTokens[loopCounter-1]);
                 toChar.println(leftSide + rightSide
                    + "  Q="+Qvalue
                    + " Type="+reacIndex
                    + " (" + reactionType[reacIndex]
                    + ")" + rDetails + "Ref:" + refString);

                 // Extract the 7 parameters from the tokens as doubles

                 for (int i=0; i<=6; i++) {
                     tokenIndex ++;
                     params[i] = stringToDouble(st.nextToken());
                 }

                 // Print 7 parameters for reaction rates

                 toChar.println( "a1="+params[0]
                    + " a2="+params[1]
                    + " a3="+params[2]
                    + " a4="+params[3]
                    + " a5="+params[4]
                    + " a6="+params[5]
                    + " a7="+params[6]);
                 toChar.println();

             }  /* end if-else loop processing of this reaction */

        }  /* end while loop on tokens */

    }  /*  End method parseBuffer  */



    // -------------------------------------------------------------------------------------------
    //  Method to convert a string to a double.  Done in two steps.
    //  First, convert the string to a Double object using the
    //  method Double.valueOf(string).  Then, extract the primitive
    //  double value of a Double object using the doubleValue()
    //  method of the Double class. (Recall that in Java the
    //  primitive number types like double and int aren't objects).
    //  Static method, so no need to instantiate.  Usage:
    //        double doubleVar = stringToDouble(myString);
    // -------------------------------------------------------------------------------------------

    static double stringToDouble (String s) {
        Double mydouble=Double.valueOf(s);    // String to Double (object)
        return mydouble.doubleValue();        // Return primitive double
    }



    // -------------------------------------------------------------------------------------
    //  Method to convert a char to a String.  Done in two steps.
    //  First, convert the char to a Character object using the
    //  Character constructor Character(c).  Then, use the
    //  toString() method of Character to return a String.
    //  Static method, so no need to instantiate.  Usage:
    //        String stringVar = charToString(myChar);
    // --------------------------------------------------------------------------------------

    static String charToString (char c) {
        Character tC = new Character(c);     // char -> Character (object)
        return tC.toString();                          // Character -> String
    }


    // -------------------------------------------------------------------------------------------------
    //  Method to return Z given the element symbol
    //  as a string (converted to lower case before comparison).  Note
    //  that there is ambiguity in Friedel table because p stands for
    //  both proton and phosphorous, and n stands for both neutron
    //  and nitrogen.  There is logic in method parseNucSymbols to
    //  take care of this.
    // -------------------------------------------------------------------------------------------------

    static int returnZ (String s) {
        int z = -1;
        s.toLowerCase();
        if (s.compareTo("n")== 0) {z=0;}
        else if (s.compareTo("p") == 0) {z=1;}
        else if (s.compareTo("d") == 0) {z=1;}
        else if (s.compareTo("t") == 0) {z=1;}
        else if (s.compareTo("he") == 0) {z=2;}
        else if (s.compareTo("li") == 0) {z=3;}
        else if (s.compareTo("be") == 0) {z=4;}
        else if (s.compareTo("b") == 0) {z=5;}
        else if (s.compareTo("c") == 0) {z=6;}
        else if (s.compareTo("n") == 0) {z=7;}     // Overridden by n above
        else if (s.compareTo("o") == 0) {z=8;}
        else if (s.compareTo("f") == 0) {z=9;}
        else if (s.compareTo("ne") == 0) {z=10;}
        else if (s.compareTo("na") == 0) {z=11;}
        else if (s.compareTo("mg") == 0) {z=12;}
        else if (s.compareTo("al") == 0) {z=13;}
        else if (s.compareTo("si") == 0) {z=14;}
        else if (s.compareTo("p") == 0) {z=15;}    // overridden by p above
        else if (s.compareTo("s") == 0) {z=16;}
        else if (s.compareTo("cl") == 0) {z=17;}
        else if (s.compareTo("ar") == 0) {z=18;}
        else if (s.compareTo("k") == 0) {z=19;}
        else if (s.compareTo("ca") == 0) {z=20;}
        else if (s.compareTo("sc") == 0) {z=21;}
        else if (s.compareTo("ti") == 0) {z=22;}
        else if (s.compareTo("v") == 0) {z=23;}
        else if (s.compareTo("cr") == 0) {z=24;}
        else if (s.compareTo("mn") == 0) {z=25;}
        else if (s.compareTo("fe") == 0) {z=26;}
        return z;
    }


    // ------------------------------------------------------------------------------------------------------
    //  Method to parse nuclear symbol in bytes and return Z and N as the
    //  two elements of the array tempInt[]
    // ------------------------------------------------------------------------------------------------------

    static int [] parseNucSymbols (byte[] b){
        int[] tempInt = new int [2];
        int length = b.length;
        int symbolLength = 1;

        // Separate the symbol from the mass number.  Ascii 0-9
        // correspond to the decimal codes 48-57.  This logic
        // determines if the second byte is a number or letter

        if (length == 1) {
            symbolLength = 1;
        } else if (length >= 2) {
            if (b[1] >= 48 && b[1] <= 57) {
                symbolLength = 1;
            } else {
                symbolLength = 2;
            }
        }

        // Generate the element symbol string

        char tC = (char)b[0];                                     // Cast 1st byte to char
        String elementSymbol = charToString(tC);    // char -> String

        if (symbolLength == 2) {                               // If symbol has 2 characters
            tC = (char)b[1];                                        // Cast 2nd byte to char
            elementSymbol += charToString(tC);        // char -> String
        }
        int Z = returnZ(elementSymbol);                   // Find Z corresponding to this symbol

        // Take care of phosphorous/proton ambiguity (both use p)
        // and nitrogen/neutron ambiguity (both use n)

        if (elementSymbol.compareTo("p") == 0 && b.length > 1) {
            Z = 15;    // This is an isotope of phosphorous, not proton
        }
        if (elementSymbol.compareTo("n") == 0 && b.length > 1) {
            Z = 7;    // This is an isotope of nitrogen, not neutron
        }

        String massString = "";
        Byte tempStuff;
        int tempInteger = 0;
        String intString = "";
        int massNumber = 0;

        // Construct string from the mass part of the symbol

        for (int i=symbolLength; i<=length-1; i++) {
            massString = Byte.toString(b[i]);
            tempStuff = Byte.decode(massString);
            tempInteger = tempStuff.intValue()-48;
            intString += Integer.toString(tempInteger);
        }
        if(intString == ""){intString = "1";}

        // Compute mass number (handle special cases first).  First
        // take care of the two states in al-26 which have the
        // notation al*6 and al-6.  Then do single symbols n,p,d,t.
        // For n/nitrogen and p/phosphorous ambiguities, resolve
        // by looking at b.length (which is 1 for proton or neutron
        // but longer for isotopes of phosphorous or nitrogen).

        if (elementSymbol.compareTo("n") == 0 && b.length == 1) {massNumber = 1;}
        else if (elementSymbol.compareTo("p") == 0 && b.length == 1) {massNumber = 1;}
        else if (elementSymbol.compareTo("d") == 0) {massNumber = 2;}
        else if (elementSymbol.compareTo("t") == 0) {massNumber = 3;}
        else {massNumber = Integer.parseInt(intString);}
        if (elementSymbol.compareTo("al") == 0) {
            if (b[2] == 42 || b[2] == 45) {massNumber = 26;}
        }

        // Return 2-d array containing Z and N

        int N = massNumber - Z;
        tempInt[0]=Z;
        tempInt[1]=N;
        return tempInt;
    }


// --------------------------------------------------------------------------------------------------------------
//  The following file copy method is adapted from a program by
//  David Flanagan to read a byte stream from one file and write it
//  intact as a byte stream to a second file (see following acknowledgement).
//  The i/o file checking and buffered input of the byte stream are
//  used almost intact from the Flanagan program. The conversion of
//  the buffer to a string, definition of the character output stream,
//  as well as the processing of the input to generate output
//  (contained primarily in the method parseBuffer()) are new.
//                        M.W. Guidry (guidry@utk.edu), Dec.28, 2001
//
// ---------------------------------------------------------------------------------------------------------------
/*
 * Copyright (c) 2000 David Flanagan.  All rights reserved.
 * This code is from the book Java Examples in a Nutshell, 2nd Edition.
 * It is provided AS-IS, WITHOUT ANY WARRANTY either expressed or implied.
 * You may study, use, and modify it for any non-commercial purpose.
 * You may distribute it non-commercially as long as you retain this notice.
 * For a commercial use license, or to purchase the book (recommended),
 * visit http://www.davidflanagan.com/javaexamples2.
 */
//package com.davidflanagan.examples.io;

/**
 * The following defines a
 * static copy() method that other programs can use to copy files.
 * Before copying the file, however, it performs a lot of tests to make
 * sure everything is as it should be.
*/
// ----------------------------------------------------------------------------

    public static void copy(String from_name, String to_name)
	throws IOException
    {
        File from_file = new File(from_name); //Get File objects from Strings
        File to_file = new File(to_name);

        // First make sure the source file exists, is a file, and is readable

        if (!from_file.exists())
            abort("no such source file: " + from_name);
        if (!from_file.isFile())
            abort("can't copy directory: " + from_name);
        if (!from_file.canRead())
            abort("source file is unreadable: " + from_name);

        // If the destination is a directory, use the source file name
        // as the destination file name

        if (to_file.isDirectory())
            to_file = new File(to_file, from_file.getName());

        // If the destination exists, make sure it is a writeable file
        // and ask before overwriting it.  If the destination doesn't
        // exist, make sure the directory exists and is writeable.

        if (to_file.exists()) {
            if (!to_file.canWrite())
                abort("destination file is unwriteable: " + to_name);

            // Ask whether to overwrite it
            System.out.print("Overwrite existing file " + to_file.getName() + "? (Y/N): ");
            System.out.flush();

            // Get the user's response.
            BufferedReader in=new BufferedReader(new InputStreamReader(System.in));
            String response = in.readLine();

            // Check the response.  If not a Yes, abort the copy.
            if (!response.equals("Y") && !response.equals("y"))
                abort("existing file was not overwritten.");
        } else {

            // If file doesn't exist, check if directory exists and is
            // writeable.  If getParent() returns null, then the directory is
            // the current dir.  so look up the user.dir system property to
            // find out what that is.

            String parent = to_file.getParent();  // Destination directory
            if (parent == null)         // If none, use the current directory
                parent = System.getProperty("user.dir");
            File dir = new File(parent);            // Convert it to a file.
            if (!dir.exists())
                abort("destination directory doesn't exist: "+parent);
            if (dir.isFile())
                abort("destination is not a directory: " + parent);
            if (!dir.canWrite())
                abort("destination directory is unwriteable: " + parent);
        }


        // Now define a byte input stream and a byte output stream,
        // and wrap the byte output stream in a character output stream.
        // The general procedure will be to read the entire file netsu
        // into a 400 kB buffer as a byte stream.  We will then manipulate
        // those bytes, often converting them to chars, Strings, ints,
        // and doubles in the process.  Finally, we will write them to
        // disk through the character output stream.

        FileInputStream from = null;  // Stream to read from source
        //FileOutputStream to = null;   // Stream to write bytes
        //PrintWriter toChar = null;    // Stream to write characters
        int buffLength = 400000;      // Length of input buffer in bytes

        // Copy the file, a buffer of bytes at a time.  For convenience,
        // we've made the buffer longer than the file netsu, so all of
        // it will be copied into the first buffer.  This will make
        // editing easier, since we don't have to worry about where the
        // streams are broken on readin.

        try {
            from = new FileInputStream(from_file);   // byte input stream
            to = new FileOutputStream(to_file);       // byte output stream
            toChar = new PrintWriter(to);                 // character output stream
                                                                        // PrintWriter takes a Writer or
                                                                        // or OutputStream as argument.
                                                                        // (PrintWriter extends Writer
                                                                        // and FileOutputStream
                                                                        // extends OutputStream.)

            byte[] buffer = new byte[buffLength];      // Buffer for file contents
            int bytes_read;                                     // How many bytes in buffer?
            buffNumber = 0;                                   // How many buffers of length
                                                                      // buffLength have been read?


            // Read bytes into the buffer, looping until we
            // reach the end of the file (when read() returns -1).

            while((bytes_read = from.read(buffer))!= -1) {  // Read til EOF

                //to.write(buffer, 0, bytes_read);       // This statment would
                                                                    // copy the input byte
                                                                    // stream to the output
                                                                    // file verbatim as a
                                                                    // byte stream.

                // Convert the input buffer to a string

                s = new String(buffer,0);
                buffNumber ++;
            }
            if(buffNumber > 1){
                callExit("\n*** Error in FriedelRead2: Buffer size "
                    +"of buffLegth="+buffLength+" exceeded for data read " 
                    +"\nfrom file "+from_file
                    +". Assign larger value for buffLength. ***");
            }
        }

        // Close the input stream, even if exceptions were thrown

        finally {
            if (from != null) try { from.close(); 
        } catch (IOException e) { ; }
        }

    } /* end method copy adapted from David Flanagan */


    /** A convenience method to throw an exception */
    private static void abort(String msg) throws IOException {
        throw new IOException("FileCopy: " + msg);
    }


    // ----------------------------------------------------------------------
    //  Method to call system exit
    // ----------------------------------------------------------------------

    public static void callExit(String message) {

        System.out.println();
        System.out.println(message);
        System.out.println();
        System.exit(1);

    }

}  /*  End class FriedelRead2  */
