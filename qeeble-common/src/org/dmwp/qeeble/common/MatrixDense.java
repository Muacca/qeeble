package org.dmwp.qeeble.common;

import java.util.Random;

public class MatrixDense implements Matrix {

 public static MatrixDense createEmpty(int rowSize, int columnSize) {
  double[] array = new double[rowSize * columnSize];
  for(int i = 0; i < array.length; ++i) {
   array[i] = 0.0;
  } 
  return new MatrixDense(rowSize, columnSize, array);
 }

 public static MatrixDense createUniformed(Random rng, int rowSize, int columnSize) {
  return new MatrixDense(
   rowSize, columnSize, Util.uniformedWeight(rng, rowSize, columnSize));
 }

 private int rowSize;
 private int columnSize;
 private double[] weight;
 public MatrixDense(int rowSize, int columnSize, double[] weight) {
  super();
  this.rowSize = rowSize;
  this.columnSize = columnSize;
  this.weight = weight;
 }

 @Override
 public final int columnSize() {
  return columnSize;
 }

 @Override
 public final int rowSize() {
  return rowSize;
 }

 @Override
 public final double get(int i, int j) {
  return weight[i * columnSize + j];
 }

 @Override
 public final void add(int i, int j, double v) {
  weight[i * columnSize + j] += v;
 }

 @Override
 public void add(Matrix m) throws Exception {
  if(m.rowSize() != rowSize || m.columnSize() != columnSize)throw new Exception("invalid dimension.");
  for(int i = 0; i < rowSize; ++i) {
   for(int j = 0; j < columnSize; ++j) {
    add(i, j, m.get(i, j));
   }
  }
 }

 @Override
 public void subtract(Matrix m) throws Exception {
  if(m.rowSize() != rowSize || m.columnSize() != columnSize)throw new Exception("invalid dimension.");
  for(int i = 0; i < rowSize; ++i) {
   for(int j = 0; j < columnSize; ++j) {
    add(i, j, -1 * m.get(i, j));
   }
  }
 }

 @Override
 public final void multiply(double v) {
  for(int i = 0; i < rowSize * columnSize; ++i) {
   weight[i] *= v;
  }
 }

 @Override
 public Vector innerProduct(Vector v) throws Exception {
  if(v.size() != columnSize)throw new Exception("invalid dimension.");
  double[] array = new double[rowSize];
  for(int i = 0; i < rowSize; ++i) {
   for(int j = 0; j < columnSize; ++j) {
    array[i] += get(i, j) * v.get(j);
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
    array[j] += v.get(i) * get(i, j);
   }
  }
  return VectorDense.create(array);
 }


}
