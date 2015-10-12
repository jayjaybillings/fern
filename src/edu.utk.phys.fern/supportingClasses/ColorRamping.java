
package supportingClasses;
import javax.vecmath.*;

// This class return either Color3f or float array RGB for Color Ramping 
// visualiation given maximum and minium value

public class ColorRamping
  {
  private double r,g,b;
  private float max, min;
  
  public ColorRamping(float inMin, float inMax)
    {
    r = 1.0;
    g = 1.0;
    b = 1.0;
    max = inMax;
    min = inMin;
    }
  
  public void getColor(float inV, Color3f color)
    {
    float value = inV;
    float dv;

    if(value < min)
      {
      value = min;
      }
      
    if(value > max)
      value = max;
    dv = max - min;
    
    r = 1.0;
    g = 1.0;
    b = 1.0;

    if(value < (min + 0.25 * dv))
      {
      r = 0.0;
      g = 4.0 * (value - min) / dv;
      }
    else if(value < (min + 0.5 * dv))
      {
      r =  0.0;
      b =  1.0 + 4.0 * (min + 0.25 * dv - value) / dv;
      }
    else if(value < (min + 0.75 * dv))
      {
      r =  4.0 * (value - min - 0.5 * dv) / dv;
      b =  0.0;
      }
    else
      {
      g =  1.0 + 4.0 * (min + 0.75 * dv - value) / dv;
      b =  0.0;
      }
    color.set((float)r,(float)g,(float)b);
    }
  }
