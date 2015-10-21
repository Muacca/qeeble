package org.dmwp.qeeble.mnist;

import java.awt.image.BufferedImage;
import java.io.IOException;

public class MNISTData {

 private MNISTInfo info;
 private int index;
 private int label;
 private byte[] buf;
 public MNISTData(MNISTInfo info, int index, int label, byte[] buf) {
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
 
 public byte[] getBuf() {
  return buf;
 }
 
 public int getLabel() {
  return label;
 }
 
 public BufferedImage getImage() throws IOException {
  return info.createImage(buf);
 }

 public String createFilename(String prefix, String ext) {
  StringBuilder sb = new StringBuilder();
  sb.append(prefix).append(index).append("_").append(label).append(".").append(ext);
  return sb.toString();
 }
}
