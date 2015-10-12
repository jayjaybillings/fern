
package supportingClasses;
import java.applet.Applet;
import java.awt.*;
import com.sun.j3d.utils.geometry.Box;
import javax.media.j3d.*;
import javax.vecmath.*;



public class Histogram3D
  {
  public Box filling;
  public Shape3D border;
  public Appearance borderAppearance;
  public Appearance fillingAppearance;
  public TransformGroup tg;
  public int protonNumber;
  public int neutronNumber;
  public float[] abundance;
  public float normalizationFactor;
  
  private static final Color3f red   = new Color3f(1.0f, 0.0f, 0.0f);
  private static final Color3f green = new Color3f(0.0f, 1.0f, 0.0f);
  private static final Color3f blue  = new Color3f(0.0f, 0.0f, 1.0f);
  private static final Color3f black  = new Color3f(0.0f, 0.0f, 0.0f);
  private static final Color3f yellow = new Color3f(1.0f, 1.0f, 0.0f);
  private static final Color3f white = new Color3f(1.0f, 1.0f, 1.0f);
  private static final Color3f grey = new Color3f(0.8f, 0.8f, 0.8f);
  
  private int timeIndex;
  
  public Histogram3D(int nProton, int nNeutron, float nFactor, int nTimeStep)
    {
    float shiftX, shiftY, shiftZ, maxZ;
    float fillScaleFactor;
    normalizationFactor = nFactor;
    timeIndex = 0;
    
    // Initialization
    abundance = new float[nTimeStep];
    protonNumber = nProton;
    neutronNumber = nNeutron;
    
    
    // Shift the histogram in Y axes so that it's below the surface initially
    shiftY = -normalizationFactor;
    
    // Scaling factor for the filling so that it does not overlap with border
    fillScaleFactor = 0.01f;
    
    // Calculate the position based on neutron and proton number
    shiftX = (float)nNeutron;
    shiftZ = -(float)nProton;
    
    
    // Create Border Box
    
    ColoringAttributes ca = new ColoringAttributes();
    ca.setColor(white);
    PolygonAttributes pa = new PolygonAttributes();
    pa.setPolygonMode(PolygonAttributes.POLYGON_LINE);
    LineAttributes la = new LineAttributes();
    la.setLineWidth(1.0f);
    
    borderAppearance = new Appearance();
    borderAppearance.setColoringAttributes(ca);
    borderAppearance.setPolygonAttributes(pa);
    borderAppearance.setLineAttributes(la);
    
    BoxFrame borderBoxFrame = new BoxFrame(0.5f, normalizationFactor, 0.5f);
    border = new Shape3D(borderBoxFrame, borderAppearance); 
    
    
    // Now Create Filled Box
    ColoringAttributes ca2 = new ColoringAttributes();
    ca2.setCapability(ColoringAttributes.ALLOW_COLOR_WRITE);
    ca2.setCapability(ColoringAttributes.ALLOW_COLOR_READ);
    ca2.setColor(red);
    
    fillingAppearance = new Appearance();
    fillingAppearance.setCapability(Appearance.ALLOW_COLORING_ATTRIBUTES_WRITE);
    fillingAppearance.setCapability(Appearance.ALLOW_COLORING_ATTRIBUTES_READ);
    fillingAppearance.setColoringAttributes(ca2);
	
	// ni's input for transparency
	  TransparencyAttributes ta = new TransparencyAttributes();
	  ta.setTransparency (0.1f);
	  ta.setTransparencyMode(TransparencyAttributes.NICEST);
	  fillingAppearance.setTransparencyAttributes(ta);
    //
    filling  = new Box(0.5f-fillScaleFactor, normalizationFactor-fillScaleFactor, 
                              0.5f-fillScaleFactor, fillingAppearance);
    
    Transform3D shiftT3 = new Transform3D();
    Vector3d shiftVector = new Vector3d(shiftX, shiftY, shiftZ);
    shiftT3.set(shiftVector);
    
    tg = new TransformGroup(shiftT3);
    tg.addChild(filling);
    tg.addChild(border);
    }
  
  public float getCurrAbundance()
    {
    return abundance[timeIndex];
    }
  
  public float getNextAbundance()
    {
    timeIndex++;
    
    if(timeIndex >= abundance.length)
      {
      timeIndex = 0;
      System.out.println("Reset");
      }
    
    return abundance[timeIndex];
    }
  
  public float getNextAbundance(int offsetIndex)
    {
    timeIndex = timeIndex + 1 + offsetIndex;
    
    if (timeIndex >= abundance.length)
      timeIndex -= abundance.length;
  
    return abundance[timeIndex];
    }
  
  
  public float getPrevAbundance()
    {
    timeIndex--;
    
    if(timeIndex < 0)
      timeIndex = abundance.length - 1;
    
    return abundance[timeIndex];
    }
  
  public float getPrevAbundance(int offsetIndex)
    {
    timeIndex = timeIndex - 1 + offsetIndex;
    
    if (timeIndex < 0)
      timeIndex += abundance.length;
  
    return abundance[timeIndex];
    }
  
  public int getCurrTimeIndex()
    {
    return timeIndex;
    }
  }
