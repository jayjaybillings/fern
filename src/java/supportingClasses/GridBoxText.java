
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
import java.awt.Font;

/*
Class GridBox builds grids in XZ plane of a box of dimensions
x times y times z.  x dimension of the box is divided into Nx 
parts, and z dimension of the box--into Nz parts.
To check the class, RotGridBox class can be executed.
GridBoxClass calls class ColorBoxLineZ (to construct a box such
that its faces are placed into the coordinate planes X, Y, Z).
Classes ColorBoxLineGridX and ColorBoxLineGridZ are called to 
build lines parallel to X axis and to Z axis correspondingly.
The  GridBox is rotated in RotGridBox class so that the XZ 
plane can be distinctly observed.
*/
public class GridBoxText extends Applet 
  {
  private int Nx, Ny, Nz;

  public GridBoxText(int myNx, int myNy, int myNz) 
    {
    Nx = myNx;
    Ny = myNy;
    Nz = myNz;
    }

    public TransformGroup createGridBoxText() 
      {
      TransformGroup objRoot = new TransformGroup();
      int NX = 2*(Nx - 1);
      int NZ = 2*(Nz - 1);

      // Constructing Box
      LineArray myBox = new ColorBoxLineZ(Nx ,Ny, Nz);
      Appearance a = new Appearance();
      Color3f red = new Color3f(1.0f, 0.0f, 0.0f);
      Color3f gray = new Color3f(0.8f, 0.8f, 0.8f);
      ColoringAttributes ca = new ColoringAttributes();
      ca.setColor(gray);
      a.setColoringAttributes(ca);
	  // after putting transparency to histogram grid became transparent also found out that antialiasing is connected to transparency; set antialiasing to false in line attributtes; also set LA to dash
      LineAttributes la = new LineAttributes(0.3f, LineAttributes.PATTERN_DASH, false);
      a.setLineAttributes(la);
      Shape3D boxShape1 = new Shape3D(myBox,a);
      Transform3D trans_static = new Transform3D();
      TransformGroup rect_static1 = new TransformGroup(trans_static);
      rect_static1.addChild(boxShape1);
      objRoot.addChild(rect_static1);

    // Preparation of text for display
      int rslt0 = 0;
      Font3D font3d = new Font3D(new Font("Arial", Font.PLAIN, 1),
                      new FontExtrusion());
      for (int n = 0; n <= 2*Nx - 1; n = n+2)
        {
        int rslt1 = rslt0 + (n/2);
        String strRslt1 = new String();
        strRslt1 = strRslt1.valueOf(rslt1);
        Text3D textGeom1 = new Text3D(font3d, strRslt1,
         new Point3f(0.0f, 0.0f, 0.0f), Text3D.ALIGN_CENTER, Text3D.PATH_RIGHT);
        Shape3D textShape1 = new Shape3D(textGeom1);
        Transform3D transform3d1 = new Transform3D();
        transform3d1.setTranslation(new Vector3f(rslt1+0.5f, 0.0f, 0.0f));
        transform3d1.setScale(0.5);
        TransformGroup contentTG1 = new TransformGroup(transform3d1);
        contentTG1.addChild(textShape1);
        objRoot.addChild(contentTG1);
        }

    // Constructing X Grid  
      LineArray myGridX = new ColorBoxLineGridX(NX,Nz);
      Appearance aGX = new Appearance();
      ColoringAttributes caGX = new ColoringAttributes();
      caGX.setColor(red);
      aGX.setColoringAttributes(caGX);
      Shape3D boxShape1GX = new Shape3D(myGridX,aGX);
      Transform3D trans_staticGX = new Transform3D();
      TransformGroup rect_static1GX = new TransformGroup(trans_staticGX);

      rect_static1GX.addChild(boxShape1GX);
      objRoot.addChild(rect_static1GX);

      for (int n = 0; n <= 2*Nz - 1; n = n+2)
        {
        int rslt1 = rslt0 - (n/2);
        String strRslt1 = new String();
        strRslt1 = strRslt1.valueOf(-rslt1);
        Text3D textGeom1 = new Text3D(font3d, strRslt1,
         new Point3f(0.0f, 0.0f, 0.0f), Text3D.ALIGN_CENTER, Text3D.PATH_RIGHT);
        Shape3D textShape1 = new Shape3D(textGeom1);
        Transform3D transform3d1 = new Transform3D();
        transform3d1.setTranslation(new Vector3f(0.0f, 0.0f, rslt1-0.5f));
        transform3d1.setScale(0.5);
        TransformGroup contentTG1 = new TransformGroup(transform3d1);
        contentTG1.addChild(textShape1);
        objRoot.addChild(contentTG1);
        }

    // Constructing Z Grid  
      LineArray myGridZ = new ColorBoxLineGridZ(NZ,Nx); // GridZ
      Appearance aGZ = new Appearance();
      ColoringAttributes caGZ = new ColoringAttributes();
      caGZ.setColor(red);
      aGZ.setColoringAttributes(caGZ);
      Shape3D boxShape1GZ = new Shape3D(myGridZ,aGZ);  // GridZ
      Transform3D trans_staticGZ = new Transform3D();
      TransformGroup rect_static1GZ = new TransformGroup(trans_staticGZ);
      rect_static1GZ.addChild(boxShape1GZ);  // GridZ
      objRoot.addChild(rect_static1GZ);

      /*
      rslt0 = 0;
      for (int n = 0; n <= 2*Ny; n = n+2)
        {
        int rslt1 = rslt0 + (n/2);
        String strRslt1 = new String();
        strRslt1 = strRslt1.valueOf(rslt1);
        Text3D textGeom1 = new Text3D(font3d, strRslt1,
         new Point3f(0.0f, 0.0f, 0.0f), Text3D.ALIGN_CENTER, Text3D.PATH_RIGHT);
        Shape3D textShape1 = new Shape3D(textGeom1);
        Transform3D transform3d1 = new Transform3D();
        transform3d1.setTranslation(new Vector3f(0.0f, rslt1, 0.0f));
        transform3d1.setScale(0.5);
        TransformGroup contentTG1 = new TransformGroup(transform3d1);
        contentTG1.addChild(textShape1);
        objRoot.addChild(contentTG1);
        }
      */
      
      return objRoot;

      }
  } // end of class GridBoxText




