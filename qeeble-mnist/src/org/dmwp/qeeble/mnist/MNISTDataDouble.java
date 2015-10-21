package org.dmwp.qeeble.mnist;

public class MNISTDataDouble {

 private MNISTInfo info;
 private int index;
 private int label;
 private double[] buf;
 public MNISTDataDouble(MNISTInfo info, int index, int label, double[] buf) {
  super();
  this.info = info;
  this.index = index;
  this.label = label;
  this.buf = buf;
 }
 
 public MNISTInfo getInfo() {
  return info;
 }
 
 public int getIndex() {
  return index;
 }
 
 public double[] getBuf() {
  return buf;
 }
 
 public int getLabel() {
  return label;
 }
 
 public String createFilename(String prefix, String ext) {
  StringBuilder sb = new StringBuilder();
  sb.append(prefix).append(index).append("_").append(label).append(".").append(ext);
  return sb.toString();
 }
}
