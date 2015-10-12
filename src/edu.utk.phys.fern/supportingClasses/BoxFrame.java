
package supportingClasses;
import javax.media.j3d.*;
import javax.vecmath.*;

class BoxFrame extends LineArray 
  {
     
  private static float x; 
  private static float y; 
  private static float z; 


  BoxFrame (float xdim, float ydim, float zdim) 
    {
    super(24, LineArray.COORDINATES);

    x = xdim;
    y = ydim;
    z = zdim;

    final float[] verts = 
      {
  	  // front face
      x, -y,  z,
      x,  y,  z,
      -x,  y,  z,
      -x, -y,  z,

  	  // back face
      -x, -y, -z,
      -x,  y, -z,
      x,  y, -z,
      x, -y, -z,

  	  // right face
      x, -y, -z,
      -x, -y, -z,
      x, -y,  z,
      -x, -y,  z,

  	  // left face
      x,  y,  z,
      -x,  y,  z,
      x,  y, -z,
      -x,  y, -z,

  	  // top face
      x,  y,  z,
      x,  y, -z,
      -x,  y, -z,
      -x,  y,  z,

  	  // bottom face
      -x, -y,  z,
      -x, -y, -z,
      x, -y, -z,
      x, -y,  z,
      };
	  setCoordinates(0, verts);
    }
  
  }
