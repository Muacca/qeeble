package org.dmwp.qeeble.mnist;

import java.awt.Point;
import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;

public class MNISTInfo {

 public static final int MAGIC_NUMBER_LABEL = 2049;
 public static final int MAGIC_NUMBER_IMAGE = 2051;
 
 public static boolean isMagicNumberLabel(int number) {
  return (number == MAGIC_NUMBER_LABEL);
 }
 
 public static boolean isMagicNumberImage(int number) {
  return (number == MAGIC_NUMBER_IMAGE);
 }
 
 private int size;
 private int rows;
 private int cols;
 public MNISTInfo(int size, int cols, int rows) {
  super();
  this.size = size;
  this.cols = cols;
  this.rows = rows;
 }

 public int getSize() {
  return size;
 }

 public int getRows() {
  return rows;
 }

 public int getCols() {
  return cols;
 }

 public byte[] createByteBuffer() {
  return new byte[rows * cols];
 }
 
 public double[] createDoubleBuffer() {
  return new double[rows * cols];
 }

 public BufferedImage createImage(byte[] data){
  return new BufferedImage(
   new ComponentColorModel(ColorSpace.getInstance(ColorSpace.CS_GRAY),
    false, false, Transparency.OPAQUE, DataBuffer.TYPE_BYTE),
   Raster.createInterleavedRaster(new DataBufferByte(data, data.length),
    cols, rows, cols, 1, new int[1], new Point()),
   false, null);
 }
 

}
