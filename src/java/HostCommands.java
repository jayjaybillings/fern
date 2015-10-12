// ----------------------------------------------------------------------------------------------------------
//  Class to read host operating system information and execute host
//  operating system commands.  Based on example at 
//  http://java.sun.com/developer/JDCTechTips/2003/tt0304.html
//  of using the .exec(String s) command to execute a system command 
//  and receive output back.  Because this class issues commands
//  that may be specific to particular operating systems, it can break
//  the Java portability.
// ----------------------------------------------------------------------------------------------------------
        
import java.io.*;
import java.util.*;
    
public class HostCommands {

    // Empty constructor

    public HostCommands(){

    } 

    // Static method to return hostname of the machine

    static String getHostname() throws IOException {
                    
        // Start the system command 

        Runtime runtime = Runtime.getRuntime();
        String argy = "hostname";
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
            allOfIt += line;
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


    // -----------------------------------------------------------------------------------------------------------------------------
    // Static method to issue general machine command and receive any output back as a
    // (possibly multiline) string. Setting the argument plines=true causes each line to be
    // printed to screen as it is produced by the command.
    // -----------------------------------------------------------------------------------------------------------------------------

    static String issueMachineCommand(String comm, Boolean plines) throws IOException {
                    
        // Start the system command 

        Runtime runtime = Runtime.getRuntime();
        Process proc = runtime.exec(comm);

        // Put a BufferedReader on the system command output

        InputStream inputstream =proc.getInputStream();
        InputStreamReader inputstreamreader = new InputStreamReader(inputstream);
        BufferedReader bufferedreader = new BufferedReader(inputstreamreader);

        // Read the system command output

        String allOfIt = ""; 
        String line;
        int lineCounter = 0;
        while ((line = bufferedreader.readLine()) != null) {
            if(plines) System.out.println(line);
            lineCounter ++;
            allOfIt += line;
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
    
    
    
    // -------------------------------------------------------------------------------------------------------------------------------------------
    // Static method to issue command to execute compiled network program, receive the
    // corresponding screen output back as a buffered stream, and both display the stream
    // line by line as it comes in and process it for information. The is essentially the
    // method issueMachineCommand customized for this particular task. The argument 'comm'
    // is the system command to be executed and the boolean 'plines' controls whether standard
    // out from the command should be displayed to the screen in unprocessed form.
    // -------------------------------------------------------------------------------------------------------------------------------------------


    static void processCompiledOutput(String comm, Boolean plines) throws IOException {
                    
        // Execute the system command by creating an instance of Runtime and using its
        // exec() method to create a Process corresponding to the command 'comm'.

        Runtime runtime = Runtime.getRuntime();
        Process proc = runtime.exec(comm);

        // Put a BufferedReader on the system command output

        InputStream inputstream =proc.getInputStream();
        InputStreamReader inputstreamreader = new InputStreamReader(inputstream);
        BufferedReader bufferedreader = new BufferedReader(inputstreamreader);

        // Read the system command output

        String line;
        int lineCounter = 0;
        while ((line = bufferedreader.readLine()) != null) {
            if(plines) System.out.println(line);              // Print stream to screen in raw form
            processLine(line);                                     // Process stream for data
            lineCounter ++;
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
    }
    
    
    // --------------------------------------------------------------------------------------------------------------------------------
    // Method to process lines coming from compiled code output in processCompiledOutput.
    // --------------------------------------------------------------------------------------------------------------------------------


    static void processLine(String line) {
    
        int index = 0;
        int index2 = 0;
        int index3 = 0;
        int index4 = 0;
        int index5 = 0;
        int index6 = 0;
        double time;
        double dt;
        double T9;
        double sumX;
        
        line = line.trim();
        
        // Extract the time,  timestep, temperature, and sumX
        
        index = line.indexOf("dt=");
        if(index != -1) {  
            index2 = line.indexOf("t=");
            index3 = line.indexOf("sdot");
            index4 = line.indexOf("T9=");
            index5 = line.indexOf("rho=");
            index6 = line.indexOf("sumX=");
            time = stringToDouble(line.substring(index2+2, index-1).trim());
            dt = stringToDouble(line.substring(index+3, index3-1).trim());
            T9 = stringToDouble(line.substring(index4+3, index5-1).trim());
            sumX = stringToDouble(line.substring(index6+5).trim());
            
            // Update the progress meters
            
            SegreFrame.prom.sets1("t="+StochasticElements.gg.decimalPlace(4,time)
                    +"  dt="+StochasticElements.gg.decimalPlace(4,dt));
                    
            SegreFrame.prom.sets2("T9="+StochasticElements.gg.decimalPlace(3,T9)
                    +"  sumX="+StochasticElements.gg.decimalPlace(4,sumX));      
        }
    
    }
    
    
    // -------------------------------------------------------------------------------------------------------
    //  Method stringToDouble to convert a string to a double.  
    // -------------------------------------------------------------------------------------------------------

    static double stringToDouble (String s) {
        Double mydouble=Double.valueOf(s);    // String to Double (object)
        return mydouble.doubleValue();              // Return primitive double
    }

}
