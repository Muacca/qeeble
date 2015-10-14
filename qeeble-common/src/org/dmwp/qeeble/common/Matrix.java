package org.dmwp.qeeble.common;

public interface Matrix {

 public int columnSize();
 public int rowSize();
 public double get(int i, int j);
 
 public void add(int i, int j, double v);
 public void add(Matrix m) throws Exception;
 public void subtract(Matrix m) throws Exception;
 public void multiply(double v);
 public Vector innerProduct(Vector v) throws Exception;
 public Vector innerProductFrom(Vector v) throws Exception;

}
