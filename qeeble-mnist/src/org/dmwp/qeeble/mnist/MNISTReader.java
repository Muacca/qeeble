package org.dmwp.qeeble.mnist;

import java.io.DataInputStream;
import java.io.IOException;

public class MNISTReader {

 public static MNISTReader create(DataInputStream labelInput, DataInputStream imageInput) throws IOException {
   if(!MNISTInfo.isMagicNumberLabel(labelInput.readInt())) {
    throw new IOException("Label file has wrong magic number(should be 2049).");
   }
   if(!MNISTInfo.isMagicNumberImage(imageInput.readInt())) {
    throw new IOException("Image file has wrong magic number(should be 2051).");
   }
   int numLabels = labelInput.readInt();
   int numImages = imageInput.readInt();
   if(numLabels != numImages) {
    throw new IOException("Image file and label file do not contain the same number of entries. Label file contains: " + numLabels + ", Image file contains: " + numImages);
   }
   return new MNISTReader(new MNISTInfo(numLabels, imageInput.readInt(), imageInput.readInt()), labelInput, imageInput);
 }

 private MNISTInfo info;
 private DataInputStream labelInput;
 private DataInputStream imageInput;
 private int count;
 private MNISTReader(MNISTInfo info, DataInputStream labelInput, DataInputStream imageInput) {
  super();
  this.info = info;
  this.labelInput = labelInput;
  this.imageInput = imageInput;
  this.count = 0;
 }
 
 public int readCount() {
  return count;
 }
 
 public boolean available() throws IOException {
  return (labelInput.available() > 0 && count < info.getSize());
 }
 
 public MNISTInfo getInfo() {
  return info;
 }

 public MNISTData read(byte[] buf) throws IOException {
  int label = labelInput.readByte();
  for(int i = 0; i < buf.length; ++i) {
   buf[i] = (byte)imageInput.readUnsignedByte();
  }
  return new MNISTData(info, count++, label, buf);
 }
 
 public MNISTData read() throws IOException {
  byte[] buf = info.createByteBuffer();
  return read(buf);
 }
 
 public MNISTDataDouble readDouble(double[] buf) throws IOException {
  int label = labelInput.readByte();
  for(int i = 0; i < buf.length; ++i) {
   buf[i] = imageInput.readUnsignedByte();
  }
  return new MNISTDataDouble(info, count++, label, buf);
 }
 
 public MNISTDataDouble readDouble() throws IOException {
  double[] buf = info.createDoubleBuffer();
  return readDouble(buf);
 }
 

}
