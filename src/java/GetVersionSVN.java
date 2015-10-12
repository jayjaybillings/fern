// ------------------------------------------------------------------------------------------------------
//  Class to read the subversion SVN version. Based on example at 
//  http://java.sun.com/developer/JDCTechTips/2003/tt0304.html
//  of using the .exec(String s) command to execute a system command 
//  and receive output back.
// ------------------------------------------------------------------------------------------------------
        
import java.io.*;
    
public class GetVersionSVN {

    static String ffname;

    static String getTheVersion(String fname, int lino) throws IOException {
    
        ffname = fname;
                    
        // Start the system command 

        Runtime runtime = Runtime.getRuntime();
        String argy = "svn info" +" " + fname;
        Process proc = runtime.exec(argy);

        // Put a BufferedReader on the system command output

        InputStream inputstream =proc.getInputStream();
        InputStreamReader inputstreamreader = new InputStreamReader(inputstream);
        BufferedReader bufferedreader = new BufferedReader(inputstreamreader);

        // Read the system command output

        String allOfIt = ""; 
        String line;
        int lineCounter = 0;
        while ((line = bufferedreader.readLine()) != null) {
            //System.out.println(line);
            lineCounter ++;
            if(lineCounter==lino) allOfIt += line;
        }
    
        // Check for system command failure
    
        try {
            if (proc.waitFor() != 0) {
                System.out.println("exit value = " + proc.exitValue());
            }
        }
        catch (InterruptedException e) {
            System.out.println(e);
        }
        return allOfIt;
    }

    public GetVersionSVN(){} 

    
/*
    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.out.println("Missing file or directory argument");
            System.exit(1);
        }
        System.out.println("Version "+ffname+" = "+getTheVersion(args[0],6));
    }
*/

}
