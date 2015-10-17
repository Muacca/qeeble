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

 private int rowSize;
 private int columnSize;
 private double[][] weight;
 private MatrixDense(int rowSize, int columnSize, double[][] weight) {
  super();
  this.rowSize = rowSize;
  this.columnSize = columnSize;
  this.weight = weight;
 }
 
 public double get(int i, int j) {
  return weight[i][j];
 }
 
 public void add(int i, int j, double v) {
  weight[i][j] += v;
 }

 public int columnSize() {
  return columnSize;
 }
 
 public int rowSize() {
  return rowSize;
 }
}
