//------------------------------------------------ 
//Program objective:
// read Data and also
// crop Data by making nonZeroAbundance for specified 
//values true only
//--------------------------------------------------

package supportingClasses;
import java.io.*;
import java.awt.*;
import java.util.*;

public class ReadFile
  {

  private static String s ; 
  public int count=0;
  public int TotaltimeStep =0;
  public double maxAbundance;
  public double minAbundance;
  public int maxProton;
  public int maxNeutron;
  public int totalTimeStep;
  public int nonZeroAbundanceCount;
  public double [][][] population;
  public double [] time; 
  public static boolean[][] nonZeroAbundance;
  public int protonValue;
  public int neutronValue;
  public int protonRangend;
  public int neutronRangend;
  
   /**************************************************** 
  // initial method for reading data////////////////////
  ******************************************************/
  
  public ReadFile (String filename)//throws IOException
    {
    int bufflength=5000000;
    int count=0;
    
    minAbundance = 1.0;
 
    try
      {
      File from_file = new File (filename);
      FileInputStream fis = new FileInputStream (from_file);
	
      //BufferedInputStream b = new BufferedInputStream(fis);
      byte[] buffer = new byte [bufflength];
      int input;
	   
      // Read data into program using loop
      while((input = fis.read(buffer))!=-1)
        { 
        s = new String (buffer);
        }
      fis.close();
      } 
	 
	 
    catch(IOException e)
      {
      System.out.println ("Error .. "+ e.toString());
      }
 
    StringTokenizer st = new StringTokenizer(s.trim());
    
    while (st.hasMoreTokens())
      {
      int step = Integer.parseInt(st.nextToken());
      double realtime =Double.parseDouble(st.nextToken());
      int Z= Integer.parseInt(st.nextToken());
      int N = Integer.parseInt(st.nextToken());
      double abundance =Double.parseDouble(st.nextToken());
		 
      // get maxAbundance 
      if(abundance > 0)
        {
        if(abundance > maxAbundance)
          {
          maxAbundance = abundance;
          }
        if(abundance < minAbundance || minAbundance == -1.0)
          {
          minAbundance = abundance;
          }
        }	 
      // get maxProton
      if(Z > maxProton)
        {
        maxProton = Z;
        }
      //get maxNeutron
      if(N > maxNeutron)
        {
        maxNeutron = N;
        }
      // get total timestep    
      if(step > totalTimeStep)
        {
        totalTimeStep = step;
        } 
      }
    // increase totalTimestep by one since step starts at zero
    totalTimeStep++;

    //initialize length of arrays from max values  
    time = new double[totalTimeStep]; 
    population = new double [maxProton+1][maxNeutron+1][totalTimeStep];
    nonZeroAbundance = new boolean [maxProton+1][maxNeutron+1];

    // define new stringtokeizer to get array data
    StringTokenizer st1 = new StringTokenizer(s.trim());
    while (st1.hasMoreTokens())
      {
      int timestep = Integer.parseInt(st1.nextToken());
      double realtime1 =Double.parseDouble(st1.nextToken());
      int protonNumber = Integer.parseInt(st1.nextToken());
      int neutronNumber =Integer.parseInt(st1.nextToken());
      double abundance1 = Double.parseDouble(st1.nextToken());
      population[protonNumber][neutronNumber][timestep] =abundance1;
      // get array for timestep      
      if(count<= timestep & count< totalTimeStep)
        {   
        time [timestep]= realtime1; 
        count++;
        }	
      }
    getNonZeroAbundance();
    }
  
  private void getNonZeroAbundance()
    {
    nonZeroAbundanceCount = 0;
    
    // Set nonZeroAbundance to false
    for(int i=0; i<=maxProton; i++)
      {
      for(int j=0; j<=maxNeutron; j++)
        {
        nonZeroAbundance[i][j] = false;
        }
      }
        
    // Traverse through the time index, then nProton, then nNeutron. 
    // Once we find an occurence which the abundance is nonzero, create a histogram for that nuclei
    
    for(int i=0; i<totalTimeStep; i++)
      {
      for(int j=0; j<=maxProton; j++)
        {
        for(int k=0; k<=maxNeutron; k++)
          {
          if(population[j][k][i] > 0)
            {
            if(nonZeroAbundance[j][k] == false)
              nonZeroAbundanceCount++;
            nonZeroAbundance[j][k] = true;
            }
          }
        }
      }
    }
	
   /************************************************************ 
  // Another method begins for croping data/////////////////////
  **************************************************************/
  
  public ReadFile ( String cropProton, String cropNeutron, String filename)//throws IOException 
    {
 
    int bufflength=5000000;
    int count=0;
    minAbundance = 1.0;
	
 
    try
      {
      File from_file = new File (filename);
      FileInputStream fis = new FileInputStream (from_file);
	
      //BufferedInputStream b = new BufferedInputStream(fis);
      byte[] buffer = new byte [bufflength];
      int input;
	   
      // Read data into program using loop
      while((input = fis.read(buffer))!=-1)
        { 
        s = new String (buffer);
        }
      fis.close();
      } 
	 
	 
    catch(IOException e)
      {
      System.out.println ("Error .. "+ e.toString());
      }
 
    StringTokenizer st = new StringTokenizer(s.trim());
    
    while (st.hasMoreTokens())
      {
      int step = Integer.parseInt(st.nextToken());
      double realtime =Double.parseDouble(st.nextToken());
      int Z= Integer.parseInt(st.nextToken());
      int N = Integer.parseInt(st.nextToken());
      double abundance =Double.parseDouble(st.nextToken());
		 
      // get maxAbundance 
      if(abundance > 0)
        {
        if(abundance > maxAbundance)
          {
          maxAbundance = abundance;
          }
        if(abundance < minAbundance || minAbundance == -1.0)
          {
          minAbundance = abundance;
          }
        }	 
      // get maxProton
      if(Z > maxProton)
        {
        maxProton = Z;
        }
      //get maxNeutron
      if(N > maxNeutron)
        {
        maxNeutron = N;
        }
      // get total timestep    
      if(step > totalTimeStep)
        {
        totalTimeStep = step;
        } 
      }
    // increase totalTimestep by one since step starts at zero
    totalTimeStep++;

    //initialize length of arrays from max values  
    time = new double[totalTimeStep]; 
    population = new double [maxProton+1][maxNeutron+1][totalTimeStep];
    nonZeroAbundance = new boolean [maxProton+1][maxNeutron+1];

    // define new stringtokeizer to get array data
    StringTokenizer st1 = new StringTokenizer(s.trim());
    while (st1.hasMoreTokens())
      {
      int timestep = Integer.parseInt(st1.nextToken());
      double realtime1 =Double.parseDouble(st1.nextToken());
      int protonNumber = Integer.parseInt(st1.nextToken());
      int neutronNumber =Integer.parseInt(st1.nextToken());
      double abundance1 = Double.parseDouble(st1.nextToken());
      population[protonNumber][neutronNumber][timestep] =abundance1;
      // get array for timestep      
      if(count<= timestep & count< totalTimeStep)
        {   
        time [timestep]= realtime1; 
        count++;
        }	
      }
 
    nonZeroAbundanceCount = 0;
    
    // Set nonZeroAbundance to false
    for(int i=0; i<=maxProton; i++)
      {
      for(int j=0; j<=maxNeutron; j++)
        {
        nonZeroAbundance[i][j] = false;
        }
      }
        
    // Traverse through the time index, then nProton, then nNeutron. 
    // using the techniques above (first method) but now adding the constranst of protonValue and neutronValue. Note: values are written as text to cropProton~Neutron; values must be entered in the order of [x1,x2,...,(xn - xm),...] note that (xn - xm) is a range of values and note the format
	
    StringTokenizer ST = new StringTokenizer(cropProton, " ,()[]");
	StringTokenizer ST2 = new StringTokenizer(cropNeutron, " ,()[]");
    while (ST.hasMoreTokens())
	  { 
	  String sg = ST.nextToken();
	  String sg2= ST2.nextToken(); 
	  char[] ch = sg.toCharArray();
	  char[] ch2 = sg2.toCharArray(); 
	  if(Character.isDigit(ch[0]) & Character.isDigit(ch2[0]))
	    {
	     protonValue = Integer.parseInt(sg);
		 neutronValue = Integer.parseInt(sg2);
		 //System.out.println (neutronValue);
		 for(int i=0; i<totalTimeStep; i++)
           {
           for(int j=0; j<=maxProton; j++)
             {
             for(int k=0; k<=maxNeutron; k++)
               {
               if(population[j][k][i] > 0 & j==protonValue & k==neutronValue)
                {
                if(nonZeroAbundance[j][k] == false)
                  nonZeroAbundanceCount++;
                  nonZeroAbundance[j][k] = true;
			    //System.out.println (population[j][k][i]);
	            }
               }
             }
            }
	      }
// if a range is given such (xn - xm) range is abudances is derived in this below commands. xn increases till it gets to xm and while it increase its abundances are recorded.
	  else
	    {
		protonRangend = Integer.parseInt(ST.nextToken());
		neutronRangend = Integer.parseInt(ST2.nextToken());
		
		for(int protonrange = protonValue+1; protonrange <= protonRangend; protonrange++)
		  {
		  for(int neutronrange = neutronValue+1 ; neutronrange <= neutronRangend; neutronrange++)
		    {
			
		    for(int i=0; i<totalTimeStep; i++)
              {
              for(int j=0; j<=maxProton; j++)
                {
                for(int k=0; k<=maxNeutron; k++)
                  {
                  if(population[j][k][i] > 0 & j==protonrange & k==neutronrange)
                    {
                    if(nonZeroAbundance[j][k] == false)
                      nonZeroAbundanceCount++;
                      nonZeroAbundance[j][k] = true;
			    //System.out.println (population[j][k][i]);
	                }  
                  }
                }
              }
		    }
		  }
		} 
     
     }   
   }
  }      
