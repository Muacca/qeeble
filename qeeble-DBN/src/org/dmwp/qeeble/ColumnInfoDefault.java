package org.dmwp.qeeble;

public class ColumnInfoDefault implements ColumnInfo {

 private int size;
 public ColumnInfoDefault(int size) {
  super();
  this.size = size;
 }
 
 public int size() {
  return size;
 }
 
 public int index(int i) {
  return i;
 }

}
