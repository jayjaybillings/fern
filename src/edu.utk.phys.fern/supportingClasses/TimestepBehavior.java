
package supportingClasses;
import java.applet.Applet;
import java.awt.*;
import java.awt.event.*;
import com.sun.j3d.utils.geometry.Box;
import javax.media.j3d.*;
import javax.vecmath.*;
import java.util.Enumeration;



public class TimestepBehavior extends Behavior
  {
  private TransformGroup targetTG;
  private Transform3D targetT3D = new Transform3D();
  private Vector3f posVector = new Vector3f();
  private int step = 0;
  private Histogram3D targetHistogram;
  private int targetMaxProton, targetMaxNeutron;
  private ColorRamping targetColorRamp;
  private Color3f histColor;
  private ColoringAttributes ca;

  // create SimpleBehavior
  public TimestepBehavior(Histogram3D histogram, int maxProton, int maxNeutron, ColorRamping colorRamp)
    {
    targetHistogram = histogram;
    this.targetTG = targetHistogram.tg;
    targetColorRamp = colorRamp;
    targetMaxNeutron = maxNeutron;
    targetMaxProton = maxProton;
    histColor = new Color3f();
    posVector.set((float)targetHistogram.neutronNumber - ((float)maxNeutron/2.0f),
                  (float)targetHistogram.abundance[0]-targetHistogram.normalizationFactor, 
                  (float) -targetHistogram.protonNumber + ((float)maxProton/2.0f));
    }
    
  public void initialize()
    {
    // set initial wakeup condition
    this.wakeupOn(new WakeupOnAWTEvent(MouseEvent.MOUSE_CLICKED));
    }

  public void processStimulus(Enumeration criteria)
    {
    posVector.set((float)targetHistogram.neutronNumber - ((float)targetMaxNeutron/2.0f), 
                  (float)targetHistogram.abundance[step]-targetHistogram.normalizationFactor, 
                  (float) -targetHistogram.protonNumber + ((float)targetMaxProton/2.0f));
    ca = targetHistogram.fillingAppearance.getColoringAttributes();
    targetColorRamp.getColor(targetHistogram.abundance[step]-targetHistogram.normalizationFactor, histColor);
    ca.setColor(histColor);
    
    step++;
    if(step >= targetHistogram.abundance.length)
      {
      step = 0;
      }
    
    targetT3D.set(posVector);
    targetTG.setTransform(targetT3D);
    this.wakeupOn(new WakeupOnAWTEvent(MouseEvent.MOUSE_CLICKED));
    }
  }
