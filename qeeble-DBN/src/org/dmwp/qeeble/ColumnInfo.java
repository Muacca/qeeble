package org.dmwp.qeeble;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

public class ColumnInfo {

 public static ColumnInfo create(int[] indexs) {
  ColumnInfo info = new ColumnInfo();
  for(int index: indexs) {
   info.columnIndexList.add(index);
  }
  return info;
 }
 
 private List<Integer> columnIndexList;
 private ColumnInfo() {
  super();
  this.columnIndexList = new ArrayList<Integer>();
 }
 
 public int size() {
  return columnIndexList.size();
 }
 
 public int index(int i) {
  return columnIndexList.get(i);
 }

 public ListIterator<Integer> iterator() {
  return columnIndexList.listIterator();
 }
}
