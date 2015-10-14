package org.dmwp.qeeble.common;

import java.util.Random;

public class MatrixDense implements Matrix {

 public static Matrix createEmpty(int rowSize, int columnSize) {
  double[][] array = new double[rowSize][columnSize];
  for(int i = 0; i < rowSize; ++i) {
   for(int j = 0; j < columnSize; ++j) {
    array[i][j] = 0.0;
   }
  } 
  return new MatrixDense(rowSize, columnSize, array);
 }
 
 public static Matrix createUniformed(Random rng, int rowSize, int columnSize) {
  return new MatrixDense(
   rowSize, columnSize, Util.uniformedWeight(rng, rowSize, columnSize));
 }

 public static Matrix cross(Vector rows, Vector cols) {
  double[][] weight = new double[rows.size()][cols.size()];
  for(int i = 0; i < rows.size(); ++i) {
   for(int j = 0; j < cols.size(); ++j) {
    weight[i][j] = rows.get(i) * cols.get(j);
   }
  }
  return new MatrixDense(rows.size(), cols.size(), weight);
 }
 
 private int rowSize;
 private int columnSize;
 private double[][] weight;
 private MatrixDense(int rowSize, int columnSize, double[][] weight) {
  super();
  this.rowSize = rowSize;
  this.columnSize = columnSize;
  this.weight = weight;
 }
 
 @Override
 public int columnSize() {
  return columnSize;
 }
 
 @Override
 public int rowSize() {
  return rowSize;
 }

 @Override
 public double get(int i, int j) {
  return weight[i][j];
 }
 
 @Override
 public void add(int i, int j, double v) {
  weight[i][j] += v;
 }

 @Override
 public void add(Matrix m) throws Exception {
  if(m.rowSize() != rowSize || m.columnSize() != columnSize)throw new Exception("invalid dimension.");
  for(int i = 0; i < rowSize; ++i) {
   for(int j = 0; j < columnSize; ++j) {
    weight[i][j] += m.get(i, j);
   }
  }
 }

 @Override
 public void subtract(Matrix m) throws Exception {
  if(m.rowSize() != rowSize || m.columnSize() != columnSize)throw new Exception("invalid dimension.");
  for(int i = 0; i < rowSize; ++i) {
   for(int j = 0; j < columnSize; ++j) {
    weight[i][j] -= m.get(i, j);
   }
  }
 }
 
 @Override
 public void multiply(double v) {
  for(int i = 0; i < rowSize; ++i) {
   for(int j = 0; j < columnSize; ++j) {
    weight[i][j] *= v;
   }
  }
 }
 
 @Override
 public Vector innerProduct(Vector v) throws Exception {
  if(v.size() != columnSize)throw new Exception("invalid dimension.");
  double[] array = new double[rowSize];
  for(int i = 0; i < rowSize; ++i) {
   for(int j = 0; j < columnSize; ++j) {
     array[i] += weight[i][j] * v.get(j);
   }
  }
  return VectorDense.create(array);
 }

 @Override
 public Vector innerProductFrom(Vector v) throws Exception {
  if(v.size() != rowSize)throw new Exception("invalid dimension.");
  double[] array = new double[columnSize];
  for(int j = 0; j < columnSize; ++j) {
   for(int i = 0; i < rowSize; ++i) {
     array[j] += v.get(i) * weight[i][j];
   }
  }
  return VectorDense.create(array);
 }


}
