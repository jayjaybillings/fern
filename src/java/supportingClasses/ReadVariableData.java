// Classes to set up the necessary variable to be used by rateviewer3d

package supportingClasses;
import java.io.*;
import java.awt.*;
import java.util.*;

public class ReadVariableData
  {
  public double maxAbundance;
  public double minAbundance;
  public int maxProton;
  public int maxNeutron;
  public int totalTimeStep;
  public int nonZeroAbundanceCount;
  public double [][][] population;
  public double [] time; 
  public static boolean[][] nonZeroAbundance;
  
  public ReadVariableData (double [][][] inputPopulation, double [] inputTime, int inputMaxProton, int inputMaxNeutron)
    {
    minAbundance = -1.0;
    maxAbundance = -1.0;
    maxProton = inputMaxProton;
    maxNeutron = inputMaxNeutron;
    totalTimeStep = inputTime.length;
    
    
    for(int i=0; i<maxProton+1; i++)
      {
      for(int j=0; j<maxNeutron+1; j++)
        {
        for(int k=0; k<totalTimeStep; k++)
          {
          if(inputPopulation[i][j][k] > 0.0)
            {
            if(maxAbundance < inputPopulation[i][j][k] || maxAbundance == -1.0)
              maxAbundance = inputPopulation[i][j][k];
            if(minAbundance > inputPopulation[i][j][k] || minAbundance == -1.0)
              minAbundance = inputPopulation[i][j][k];
            }
          }
        }
      }
    
    System.out.println("MAX ABUNDANCE: " + maxAbundance);
    //initialize length of arrays from max values
    population = new double [maxProton+1][maxNeutron+1][totalTimeStep];
    population = inputPopulation;
    time = new double[totalTimeStep]; 
    time = inputTime;
    
    nonZeroAbundance = new boolean [maxProton+1][maxNeutron+1];
    getNonZeroAbundance();
    }
  
  
  public ReadVariableData (double [][][] inputPopulation, double [] inputTime, int inputMaxProton, int inputMaxNeutron, double lowerCutoff)
    {
    minAbundance = -1.0;
    maxAbundance = -1.0;
    maxProton = inputMaxProton;
    maxNeutron = inputMaxNeutron;
    totalTimeStep = inputTime.length;
    
    
    for(int i=0; i<maxProton+1; i++)
      {
      for(int j=0; j<maxNeutron+1; j++)
        {
        for(int k=0; k<totalTimeStep; k++)
          {
          
          // If element abundance < 1E-8, drop it to zero
          // FIX ME: This is hardcoded - bad.
          if(inputPopulation[i][j][k] < lowerCutoff)
            inputPopulation[i][j][k] = 0.0;
            
          if(inputPopulation[i][j][k] > 0.0)
            {
            if(maxAbundance < inputPopulation[i][j][k] || maxAbundance == -1.0)
              maxAbundance = inputPopulation[i][j][k];
            if(minAbundance > inputPopulation[i][j][k] || minAbundance == -1.0)
              minAbundance = inputPopulation[i][j][k];
            }
          }
        }
      }
    
    System.out.println("MAX ABUNDANCE: " + maxAbundance);
    //initialize length of arrays from max values
    population = new double [maxProton+1][maxNeutron+1][totalTimeStep];
    population = inputPopulation;
    time = new double[totalTimeStep]; 
    time = inputTime;
    
    nonZeroAbundance = new boolean [maxProton+1][maxNeutron+1];
    getNonZeroAbundance();
    }
  
  
  
  // Overload ReadVariableData Constructor to read from a file
  
  public ReadVariableData (String filename)//throws IOException
    {
    String dataLine = new String(); 
    int tmpTimeIndex, tmpProton, tmpNeutron;
    double tmpTime, tmpAbundance, lowerCutoff;
    
    lowerCutoff = 0.0;
    try
      {
      BufferedReader in = new BufferedReader(new FileReader(filename));
      
      // Asume we have header file that tells the totalTimeStep, maxProton & maxNeutron
      dataLine = in.readLine();
      StringTokenizer myToken= new StringTokenizer(dataLine, " ");
      
      totalTimeStep = Integer.parseInt(myToken.nextToken());
      maxProton = Integer.parseInt(myToken.nextToken());
      maxNeutron = Integer.parseInt(myToken.nextToken());
      if(myToken.hasMoreTokens())
        lowerCutoff = Double.parseDouble(myToken.nextToken());
      
      // Initialize Length of arrays from max values
      population = new double[maxProton+1][maxNeutron+1][totalTimeStep];
      time = new double[totalTimeStep];
      
      minAbundance = -1.0;
      maxAbundance = -1.0;
      
      // Read from file
      while((dataLine = in.readLine()) != null)
        {
        myToken = new StringTokenizer(dataLine, " ");
        tmpTimeIndex = Integer.parseInt(myToken.nextToken());
        tmpTime      = Double.parseDouble(myToken.nextToken());
        tmpProton    = Integer.parseInt(myToken.nextToken());
        tmpNeutron   = Integer.parseInt(myToken.nextToken());
        tmpAbundance = Double.parseDouble(myToken.nextToken());
        
        // If element abundance < 1E-8, drop it to zero
        // FIX ME: This is hardcoded - bad.
        if(tmpAbundance < lowerCutoff)
          tmpAbundance = 0.0;
        
        time[tmpTimeIndex] = tmpTime;
        population[tmpProton][tmpNeutron][tmpTimeIndex] = tmpAbundance;
        
        if(tmpAbundance > 0)
          {
          if(maxAbundance < tmpAbundance || maxAbundance == -1.0)
            maxAbundance = tmpAbundance;
          if(minAbundance > tmpAbundance || minAbundance == -1.0)
            minAbundance = tmpAbundance;
          }
        }
      in.close();
      }
    catch(IOException e)
      {
      System.out.println("Cannot read input file: Exception:" + e);
      }
    
    nonZeroAbundance = new boolean[maxProton+1][maxNeutron+1];
    getNonZeroAbundance();
    }
  
  public ReadVariableData( String cropProton, String cropNeutron, String filename)//throws IOException 
    {
    // BROKEN , FIX ME
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
	
  } // End class
