
package supportingClasses;
import java.awt.*;
import javax.vecmath.*;

public class TimeDisplay
  {
  public TextField timeText;
  private double maxTime;
  private double minTime;
  private double curTime;
  private String curTimeStr;
  
  public TimeDisplay(double minimumTime, double maximumTime)
    {
    maxTime = maximumTime;
    minTime = minimumTime;
    System.out.println("MaxTime = " + maxTime);
    timeText = new TextField(10);
    timeText.setEditable(false);
    }
  
  public TimeDisplay()
    {
    timeText = new TextField(10);
    timeText.setEditable(false);
    maxTime = 1.0;
    }
  
  public void setMinMaxTime(double minimumTime, double maximumTime)
    {
    maxTime = maximumTime;
    minTime = minimumTime;
    // System.out.println("MaxTime = " + maxTime);
    }
  
  public void updateDisplayFromAlpha(float alphaValue)
    {
    if(minTime == 0.0)
      curTime = (double) alphaValue * Math.log(maxTime);
    else
      curTime = alphaValue * (Math.log(maxTime) - Math.log(minTime)) + Math.log(minTime);
    
    // System.out.println("Alpha: " + alphaValue + " CurTime: " + curTime);  
    curTime = Math.exp(curTime);
    curTimeStr = Double.toString(curTime);
    // System.out.println("Alpha: " + alphaValue + " time: " + curTimeStr ); // + "maxTime: " + maxTime + " minTime: " + minTime );
    
    if(curTimeStr.indexOf(".") == 0 )
      curTimeStr = curTimeStr.substring(0, 4);
    else
      if(curTimeStr.indexOf("E") > 0)
        curTimeStr = curTimeStr.substring(0, curTimeStr.indexOf(".") + 4) + curTimeStr.substring(curTimeStr.indexOf("E")) ;
      else  
        curTimeStr = curTimeStr.substring(0, curTimeStr.indexOf(".") + 4);
    
    timeText.setText(curTimeStr);
    }
  
  public void updateDisplayFromTime(double curTime)
    {
    curTimeStr = Double.toString(curTime);
    // System.out.println("CurTimeStep: " + curTimeStr);
    if(curTimeStr.indexOf(".") == 0)
      curTimeStr = curTimeStr.substring(0, 6);
    else
      if(curTimeStr.indexOf("E") > 0)
        curTimeStr = curTimeStr.substring(0, curTimeStr.indexOf(".") + 6) + curTimeStr.substring(curTimeStr.indexOf("E")) ;
      else
        if(curTimeStr.length() < curTimeStr.indexOf(".") + 6) 
          curTimeStr = curTimeStr;
        else
          curTimeStr = curTimeStr.substring(0, curTimeStr.indexOf(".") + 6);
    timeText.setText(curTimeStr);
    }
  
  public double getTimeFromAlpha(float alphaValue)
    {
    curTime = (double) alphaValue * Math.log(maxTime)/Math.log(10);
    curTime = Math.pow(10.0, curTime);
    return curTime;
    }
  }
  
    
    
        
  