
import java.io.*;

class ReadAFile{

    public int bufferLength;

    // Overload the constructor so that an instance can be invoked with an argument
    // giving the size of the buffer required for that instance instead of using the class default.

    ReadAFile(){}

    ReadAFile(int bufferLength){
        this();           // Causes first constructor to be run first (not really required in this case)
        this.bufferLength = bufferLength;
    }

    // -------------------------------------------------------------------------------------------------------------
    //  The following file readin method is adapted from a program by
    //  David Flanagan.  It tests for common file input mistakes.  To use,
    //  create an instance of this class and then invoke the method
    //  readASCIIFile(filename), which returns the file content of the ascii
    //  file filename as a single string.  This string can then be broken into 
    //  tokens (separated on whitespace) using the StringTokenizer class and 
    //  the tokens processed to store the data read in as variables.  See 
    //  ReadAFileTester.java for examples.
    // -------------------------------------------------------------------------------------------------------------

    public String readASCIIFile(String from_name) throws IOException {

        String s = null;
        int buffNumber;

        File from_file = new File(from_name);   //Get File objects from Strings

        // First make sure the source file exists, is a file, and is readable

        if (!from_file.exists())
            abort("no such source file: " + from_name);
        if (!from_file.isFile())
            abort("can't copy directory: " + from_name);
        if (!from_file.canRead())
            abort("source file is unreadable: " + from_name);

        // Now define a byte input stream
        FileInputStream from = null;             // Stream to read from source
        int buffLength = 50000;//32768;//16384;           // Length of input buffer in bytes
        if(this.bufferLength > 0) buffLength=this.bufferLength;

        // Copy the file, a buffer of bytes at a time.

        try {
            from = new FileInputStream(from_file);    // byte input stream

            byte[] buffer = new byte[buffLength];         //Buffer for file contents
            int bytes_read;                                        //How many bytes in buffer?
            buffNumber = 0;                                      //How many buffers of length
                                                                         //buffLength have been read?

            // Read bytes into the buffer, looping until we
            // reach the end of the file (when read() returns -1).

            while((bytes_read = from.read(buffer))!= -1) {      // Read til EOF
                // Convert the input buffer to a string
                s = new String(buffer,0);
                buffNumber ++;
            }

            if(buffNumber > 1){
                callExit("\n*** Error in ReadAFile: Buffer size "
                    +"of buffLength="+buffLength+" exceeded for data input stream " 
                    +"\nfrom file "+from_name
                    +". Assign larger value for buffLength. ***");
            }
        }

        // Close the input stream, even if exceptions were thrown

        finally {
            if (from != null) try { from.close(); } catch (IOException e) { ; }
        }

        // Return the input buffer as a string.  This string may be broken on whitespace
        // into tokens using the StringTokenizer class and then the tokens processed into
        // data.  See ReadAFileTester.java for examples.

        return s.trim();
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


    /** A convenience method to throw an exception */
    private static void abort(String msg) throws IOException {
        throw new IOException("FileCopy: " + msg);
    }
}