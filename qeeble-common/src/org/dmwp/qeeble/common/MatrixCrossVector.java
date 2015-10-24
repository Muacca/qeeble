package org.dmwp.qeeble.common;

public class MatrixCrossVector implements Matrix {

 public static MatrixCrossVector create(Vector rows, Vector cols) {
  return new MatrixCrossVector(rows, cols);
 }
 
 private Vector rows;
 private Vector cols;
 private MatrixCrossVector(Vector rows, Vector cols) {
  super();
  this.rows = rows;
  this.cols = cols;
 }
 
 @Override
 public final int columnSize() {
  return cols.size();
 }
 
 @Override
 public final int rowSize() {
  return rows.size();
 }

 @Override
 public final double get(int i, int j) {
  return rows.get(i) * cols.get(j);
 }
 
 @Override
 public void add(int i, int j, double v) {
  //not supported;
 }

 @Override
 public void add(Matrix m) throws Exception {
  //not supported;
 }

 @Override
 public void subtract(Matrix m) throws Exception {
  //not supported;
 }
 
 @Override
 public void multiply(double v) {
  //not supported;
 }
 
 @Override
 public Vector innerProduct(Vector v) throws Exception {
  if(v.size() != cols.size())throw new Exception("invalid dimension.");
  double[] array = new double[rows.size()];
  for(int i = 0; i < rows.size(); ++i) {
   for(int j = 0; j < cols.size(); ++j) {
     array[i] += get(i, j) * v.get(j);
   }
  }
  return VectorDense.create(array);
 }

 @Override
 public Vector innerProductFrom(Vector v) throws Exception {
  if(v.size() != rows.size())throw new Exception("invalid dimension.");
  double[] array = new double[cols.size()];
  for(int j = 0; j < cols.size(); ++j) {
   for(int i = 0; i < rows.size(); ++i) {
     array[j] += v.get(i) * get(i, j);
   }
  }
  return VectorDense.create(array);
 }


}
