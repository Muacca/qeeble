package org.dmwp.qeeble.mnist;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

public class MNISTImageConvertTest {

 /**To convert MNIST file to PNG images
  * @param args
  *          args[0]: label file
  *          args[1]: image file
  *          args[2]: output dir
  * @throws IOException
  */
 public static void main(String[] args) {
  if(args.length != 3) {
   System.err.println("usage: <label file> <image file> <output dir>");
   System.exit(1);
  }
  long start = System.currentTimeMillis();
  try(
   DataInputStream labelInput = new DataInputStream(new BufferedInputStream(new FileInputStream(args[0])));
   DataInputStream imageInput = new DataInputStream(new BufferedInputStream(new FileInputStream(args[1])));
   ){
   String prefix = args[2] + "/image";
   MNISTReader in = MNISTReader.create(labelInput, imageInput);
   System.out.println("size:" + in.getInfo().getSize() + ", rows:" + in.getInfo().getRows() + ", cols:" + in.getInfo().getCols());
   byte[] buf = in.getInfo().createByteBuffer();
   while(in.available()) {
    MNISTData data = in.read(buf);
    ImageIO.write(data.getImage(), "png", new File(data.createFilename(prefix, "png")));
    if(in.readCount() % 100 == 0) {
     System.out.print(".");
     if(in.readCount() % 1000 == 0) {
      System.out.println(in.readCount() + "/" + in.getInfo().getSize() + ": " + (System.currentTimeMillis() - start) + "msec");
     }
    }
   }
  }catch(Exception e) {
   e.printStackTrace();
   System.exit(1);
  }
 }


}
