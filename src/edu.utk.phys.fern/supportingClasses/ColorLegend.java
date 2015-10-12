
package supportingClasses;

import java.applet.Applet;
import java.awt.BorderLayout;
import java.awt.Frame;
import java.awt.event.*;
import java.awt.GraphicsConfiguration;
import com.sun.j3d.utils.applet.MainFrame; 
import com.sun.j3d.utils.universe.*;
import javax.media.j3d.*;
import javax.vecmath.*;
import com.sun.j3d.utils.geometry.ColorCube;
import com.sun.j3d.utils.geometry.Cylinder;
import com.sun.j3d.utils.geometry.Box;
import java.util.Enumeration;
import java.lang.Object;
import java.awt.Font;
import com.sun.j3d.utils.geometry.Text2D;

public class ColorLegend extends Applet 
  {
  
  private TransformGroup ColorLegend;
  private float x, y, z;
  private int N;
  private float minColor, maxColor;
    
  public ColorLegend(float min, float max) 
    {
    N = 80;  // N is even for now
    x = 0.005f;
    y = 0.02f;
    z = 0.0f;
    minColor = min;
    maxColor = max;
    }
  
  public ColorLegend(float min, float max, float xstep, float height, float width, int totalStep)
    {
    x = xstep;
    y = height;
    z = width;
    N = totalStep;
    }
    
  //
  // create scene graph branch group
  //
  public TransformGroup createColorLegend() 
    {
    float step = x+x;
    float L = -(N/2)*step;
    
    float stepMM = (maxColor - minColor)/4;

    float rslt1 = minColor;
    String strRslt1 = new String();
    strRslt1 = strRslt1.valueOf(rslt1);

    float rslt2 = minColor + stepMM;
    String strRslt2 = new String();
    strRslt2 = strRslt2.valueOf(rslt2);

    float rslt3 = minColor + 2*stepMM;
    String strRslt3 = new String();
    strRslt3 = strRslt2.valueOf(rslt3);
    
    float rslt4 = minColor + 3*stepMM;
    String strRslt4 = new String();
    strRslt4 = strRslt4.valueOf(rslt4);

    float rslt5 = minColor + 4*stepMM;
    String strRslt5 = new String();
    strRslt5 = strRslt4.valueOf(rslt5);

    TransformGroup objRoot = new TransformGroup();
    Font3D font3d = new Font3D(new Font("Arial", Font.PLAIN, 1),
                            new FontExtrusion());
    Text3D textGeom1 = new Text3D(font3d, strRslt1,
        new Point3f(0.0f, 0.0f, 0.0f), Text3D.ALIGN_CENTER, Text3D.PATH_RIGHT);
    Shape3D textShape1 = new Shape3D(textGeom1);    

    Text3D textGeom2 = new Text3D(font3d, new String(strRslt5),
        new Point3f(0.0f, 0.0f, 0.0f), Text3D.ALIGN_CENTER, Text3D.PATH_RIGHT);
    Shape3D textShape2 = new Shape3D(textGeom2);    

    Text3D textGeom3 = new Text3D(font3d, new String(strRslt3),
        new Point3f(0.0f, 0.0f, 0.0f), Text3D.ALIGN_CENTER, Text3D.PATH_RIGHT);
    Shape3D textShape3 = new Shape3D(textGeom3);    

    Text3D textGeom4 = new Text3D(font3d, strRslt2,
        new Point3f(0.0f, 0.0f, 0.0f), Text3D.ALIGN_CENTER, Text3D.PATH_RIGHT);
    Shape3D textShape4 = new Shape3D(textGeom4);    

    Text3D textGeom5 = new Text3D(font3d, new String(strRslt4),
        new Point3f(0.0f, 0.0f, 0.0f), Text3D.ALIGN_CENTER, Text3D.PATH_RIGHT);
    Shape3D textShape5 = new Shape3D(textGeom5);    

    Transform3D transform3D1 = new Transform3D();
    transform3D1.setTranslation(new Vector3f(-0.05f, L, 0.0f));
    transform3D1.setScale(0.03);
    TransformGroup contentTG1 = new TransformGroup(transform3D1);
    contentTG1.addChild(textShape1);
    objRoot.addChild(contentTG1);

    Transform3D transform3D2 = new Transform3D();
    transform3D2.setTranslation(new Vector3f(-0.05f, -L, 0.0f));
    transform3D2.setScale(0.03);
    TransformGroup contentTG2 = new TransformGroup(transform3D2);
    contentTG2.addChild(textShape2);
    objRoot.addChild(contentTG2);
    Transform3D transform3D3 = new Transform3D();
    transform3D3.setTranslation(new Vector3f(-0.05f, 0.0f, -0.0f));
    transform3D3.setScale(0.03);
    TransformGroup contentTG3 = new TransformGroup(transform3D3);
    contentTG3.addChild(textShape3);
    objRoot.addChild(contentTG3);

    Transform3D transform3D4 = new Transform3D();
    transform3D4.setTranslation(new Vector3f(-0.05f, L/2.0f, 0.0f));
    transform3D4.setScale(0.03);
    TransformGroup contentTG4 = new TransformGroup(transform3D4);
    contentTG4.addChild(textShape4);
    objRoot.addChild(contentTG4);

    Transform3D transform3D5 = new Transform3D();
    transform3D5.setTranslation(new Vector3f(-0.05f, -L/2.0f, 0.0f));
    transform3D5.setScale(0.03);
    TransformGroup contentTG5 = new TransformGroup(transform3D5);
    contentTG5.addChild(textShape5);
    objRoot.addChild(contentTG5);

    ColorRamping myRamp = new ColorRamping(minColor, maxColor);
    float c0 = minColor;
    float stepColor = (maxColor - minColor)/N;
    Transform3D trans_static = new Transform3D();
    
    Transform3D allBoxesT3D = new Transform3D();
    allBoxesT3D.rotZ(Math.PI/2.0f);
    
    TransformGroup allBoxesTG = new TransformGroup();
    allBoxesTG.setTransform(allBoxesT3D);
    
    for(int n=0; n < N; ++n) 
      {
      float cn = c0 + stepColor*n;
      Appearance a = new Appearance();
      Color3f myColor = new Color3f();
      myRamp.getColor(cn, myColor);
      ColoringAttributes ca = new ColoringAttributes();
      ca.setColor(myColor);
      a.setColoringAttributes(ca);

      Box myBox = new Box(x, y, z, a);
      trans_static.setTranslation(new Vector3f(L+n*step,0.1f,0.0f));
      TransformGroup rect_static1 = new TransformGroup(trans_static);
      rect_static1.addChild(myBox);
      allBoxesTG.addChild(rect_static1);
      }
    
    
    objRoot.addChild(allBoxesTG);
    return objRoot;
    } // end of CreateSceneGraph method of rectangleTranslation

  } 
  // end of class ColorLegend
