package org.dmwp.qeeble.common;

import java.util.Random;

public class Util {

 public static double[] softmax(double[] array) {
  double max = max(array);
  double sum = 0.0;
  for(int i = 0; i < array.length; ++i) {
   array[i] = Math.exp(array[i] - max);
   sum += array[i];
  }
  for(int i = 0; i < array.length; ++i) {
   array[i] = array[i] / sum;
  }
  return array;
 }
  
 public static double max(double[] array) {
  double max = 0.0;
  for(double x: array) {
   max = Math.max(max, x);
  }
  return max;
 }
 
 public static double sigmoid(double x) {
  return 1.0 / (1.0 + Math.pow(Math.E, -1 * x));
 }
 
 public static double[] sigmoid(double[] array) {
  for(int i = 0; i < array.length; ++i) {
   array[i] = sigmoid(array[i]);
  }
  return array;
 }
 
 public static int binomial(Random rng, double p) {
  if(p < 0 || p > 1) return 0;
  if(rng.nextDouble() < p)return 1;
  return 0;
 }
 
 public static int[] binomial(Random rng, double[] array) {
  int[] b = new int[array.length];
  for(int i = 0; i < array.length; ++i) {
   b[i] = binomial(rng, array[i]);
  }
  return b;
 }
 
 public static double uniform(Random rng, double min, double max) {
  return rng.nextDouble() * (max - min) + min;
 }
 
 public static double[] uniformedWeight(Random rng, int outputSize, int inputSize) {
  double[] array = new double[outputSize * inputSize];
  double a = 1.0 / inputSize;
  for(int i = 0; i < array.length; ++i) {
   array[i] = uniform(rng, -a, a); 
  }
  return array;
 }

 public static double[] uniformArray(Random rng, int n) {
  double[] array = new double[n];
  double a = 1.0 / n;
  for(int i = 0; i < n; ++i) {
   array[i] = uniform(rng, -a, a); 
  }
  return array;
 }
 
 private Util() {}

}
