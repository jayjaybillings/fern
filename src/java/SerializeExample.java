// ----------------------------------------------------------------------------------
// Test driver for ReactionClass1 serializer.  This will
// serialize 3 examples of the class ReactionClass1 to
// the file "rc.ser".  To deserialize the file and print
// datafields of its 3 ReactionClass1 objects, run the
// class LoadR1 with rc.ser as command-line argument:
//        "java LoadR1 rc.ser"
// ----------------------------------------------------------------------------------

import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;

public class SerializeExample {

    private static Point [] In = new Point[3];
    private static Point [] Out = new Point[4];


    public static void main(String[] args) {

        ReactionClass1 [] instance = new ReactionClass1 [3];
        double [] tparms = new double[7];
        int [] numberEachType = {3,1,1,1,0,0,0,0,0};

        for (int i=0; i<7; i++) {
            tparms[i] = (double) i / 10;
        }

        nullAll();
        In[0] = new Point(6,6);
        Out[0] = new Point(5,7);

        instance[0] = new ReactionClass1(1,1,1,false,false,false,true,"a->b","ref1",
            In,Out,5.0,tparms);

        for (int i=0; i<7; i++) {
            tparms[i] = (double) i / 20;
        }

        nullAll();
        In[0] = new Point(4,4);
        Out[0] = new Point(3,5);

        instance[1] = new ReactionClass1(1,1,1,false,false,true,false,"a->b","ref2",
            In,Out,4.0,tparms);

        nullAll();
        In[0] = new Point(6,6);
        Out[0] = new Point(5,7);

        instance[2] = new ReactionClass1(1,1,1,false,false,true,false,"a->b","ref3",
            In,Out,5.0,tparms);

        // Call the serializeIt method of ReactionClass1 to serialize
        // the objects contained in the array instance[].

        ReactionClass1.serializeIt("rc.ser", numberEachType, instance);

    }

    static void nullAll() {
        In[0] = Out[0] = null;
        In[1] = Out[1] = null;
        In[2] = Out[2] = null;
        Out[3] = null;
    }

}


