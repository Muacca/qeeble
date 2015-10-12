package org.dmwp.qeeble.common;

public class ColumnInfoIndexed implements ColumnInfo {

 private int[] indexs;
 public ColumnInfoIndexed(int[] indexs) {
  super();
  this.indexs = indexs;
 }
 
 public int size() {
  return indexs.length;
 }
 
 public int index(int i) {
  return indexs[i];
 }

}
