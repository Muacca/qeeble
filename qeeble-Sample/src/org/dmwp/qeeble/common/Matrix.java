package org.dmwp.qeeble.common;

public interface Matrix {

 public double get(int i, int j);
 public void add(int i, int j, double v);
 public int columnSize();
 public int rowSize();

}
