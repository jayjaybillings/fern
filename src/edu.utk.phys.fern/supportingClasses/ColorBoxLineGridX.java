
package supportingClasses;
import javax.media.j3d.*;
import javax.vecmath.*;

class ColorBoxLineGridX extends LineArray
  {
  private static int NX;
  private static int Nz;

  ColorBoxLineGridX(int NX, int Nz)
    {
    super(NX, LineArray.COORDINATES);
    final Point3f Xgrids[] = new Point3f[NX];
    int n;
      for (n = 0; n < NX; n = n+2)
        {
        Xgrids[n] = new Point3f((n/2+1), 0.0f, 0.0f);
        Xgrids[n+1] = new Point3f((n/2+1), 0.0f,  -Nz);
        }

       setCoordinates(0, Xgrids);

     }
  } 
