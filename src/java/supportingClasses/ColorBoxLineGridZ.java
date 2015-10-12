
package supportingClasses;
import javax.media.j3d.*;
import javax.vecmath.*;

class ColorBoxLineGridZ extends LineArray 
  {
  private static int NZ;
  private static int Nx;
  ColorBoxLineGridZ(int NZ, int Nx) 
    {
    super(NZ, LineArray.COORDINATES);
    final Point3f Zgrids[] = new Point3f[NZ];
    int n;    
      for (n = 0; n < NZ; n = n+2)
        {
        Zgrids[n] = new Point3f(0.0f, 0.0f, -(n/2+1));
        Zgrids[n+1] = new Point3f(Nx, 0.0f, -(n/2+1));
        } 
       setCoordinates(0, Zgrids);

     } 
  } 
