// ------------------------------------------------------------------------------------------------------------
//  Class ListSystemProperties to list System Properties to either an
//  output file, or standard out (normally the screen).  (Streams are
//  required, because the System.getProperties.list(PrintStream out) only
//  generates output to a PrintStream.)  This class illustrates
//  the use of file output streams and writers to write to either a file
//  or standard out.  The constructor is overloaded.  If used with no
//  arguments it sends the list of properties to the screen.  If with
//  a string argument fileName, it sends the list to the file fileName.
//  There is a main method test driver to test from the command line,
//  or the constructors can be invoked directly from other classes.
//  See the comments preceding the main method for command line usage.
// -------------------------------------------------------------------------------------------------------------

import java.io.*;


public class ListSystemProperties {


    // ----------------------------------------------------------------------------------------------------------
    //  Overloaded constructors.  This constructor takes no argument and
    //  sends the listed System Properties to standard out (normally
    //  the screen).  In this case we must use a Java 1.0 PrintStream instead
    //  of a Java 1.1 PrintWriter object (as we use in the second form of the
    //  overloaded constructor below) because System.out is a PrintStream.
    // ----------------------------------------------------------------------------------------------------------

    public ListSystemProperties () {

       PrintStream npr = System.out;   // Assign to standard output

       // Write system properties to standard out (screen)

       System.out.println("System Properties:\n");
       System.getProperties().list(npr);

       npr.flush(); npr.close();       // Flush and close stream

    }



    // ----------------------------------------------------------------------------------------------------------
    //  Overloaded constructors.  This constructor takes a String filename
    //  as argument and sends the listed System Properties to the named file.
    //  In this case we may use a PrintWriter rather than a PrintStream as
    //  we had to use in the first form of the constructor.
    // ----------------------------------------------------------------------------------------------------------

    public ListSystemProperties (String fileName) {

        // Print system Properties to the file fileName.  The FileOutputStream
        // requires that IOException be caught or declared.  Let's catch it but
        // do nothing with the exception.

        try {

            FileOutputStream to = new FileOutputStream(fileName);
            PrintWriter pr = new PrintWriter(to);

            // Write System Properties to a file

            pr.println("System Properties:\n");
            System.getProperties().list(pr);

            pr.flush(); pr.close();
            to.flush(); to.close();

        } catch (IOException e) {;}

    }


    // --------------------------------------------------------------------------
    //  Main method.  Test driver for overloaded methods defined above.  If
    //  invoked from the command line without an argument:
    //
    //        java ListSystemProperties
    //
    //  the no-argument constructor is invoked and System Properties are
    //  listed to standard output (screen).  If invoked from the command line
    //  with an argument:
    //
    //        java ListSystemProperties fileName
    //
    //  the System Properties are listed to the file specified by fileName.  If
    //  more than one argument is supplied, an error message is displayed.
    // --------------------------------------------------------------------------

    public static void main(String[] args) {

        if (args.length > 1) {
            System.out.println("\nExit:  Wrong number of arguments.  Either no"
                + " arguments for a screen list\n"
                + "or one filename argument to list to file.");
            System.exit(1);
        } else if (args.length == 1) {
            ListSystemProperties lsp = new ListSystemProperties(args[0]);
        } else {
            ListSystemProperties lsp2 = new ListSystemProperties();
        }
    }

}  /* End class ListSystemProperties */