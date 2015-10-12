
package supportingClasses;
import javax.media.j3d.*;
import javax.vecmath.*;

class ColorBoxLineZ extends LineArray 
  {

  private static int x; 
  private static int y; 
  private static int z; 

  // ColorBoxLineZ (float xdim, float ydim, float zdim)
  ColorBoxLineZ (int Nx, int Ny, int Nz) 
    {
    super(24, LineArray.COORDINATES);

    x = Nx;
    y = Ny;
    z = Nz;

    final float[] verts = 
      {
      // front face
      0.0f,  0.0f,  0.0f,
      0.0f,     y,  0.0f,
         x,  0.0f,  0.0f,
         x,     y,  0.0f,
      // back face
      0.0f , 0.0f,    -z,
      0.0f,     y,    -z,
         x,  0.0f,    -z,
         x,     y,    -z,

      // right face
         x,  0.0f,  0.0f,
         x,  0.0f,    -z,
         x,     y,  0.0f,
         x,     y,    -z,

      // left face
      0.0f,  0.0f,  0.0f,
      0.0f,  0.0f,    -z,
      0.0f,     y,  0.0f,
      0.0f,     y,    -z,

      // top face
      0.0f,     y,    -z,
         x,     y,    -z,
      0.0f,     y,  0.0f,
         x,     y,  0.0f,

      // bottom face
      0.0f,  0.0f,    -z,
         x,  0.0f,    -z,
      0.0f,  0.0f,  0.0f,
         x,  0.0f,  0.0f,

       };
       setCoordinates(0, verts);
     } 
  }
