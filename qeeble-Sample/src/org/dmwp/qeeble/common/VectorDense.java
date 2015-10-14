package org.dmwp.qeeble.common;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

public class VectorDense implements Vector {
 
 public static VectorDense create(int[] array) {
  double[] temp = new double[array.length];
  for(int i = 0; i < array.length; ++i) {
   temp[i] = (double)array[i];
  }
  return new VectorDense(temp);
 }
 
 public static VectorDense create(double[] array) {
  return new VectorDense(array);
 }
 
 public static VectorDense createEmpty(int columnSize) {
  double[] array = new double[columnSize];
  Arrays.fill(array, 0.0);
  return new VectorDense(array);
 }
 

 private double[] array;
 private VectorDense(double[] array){
  super();
  this.array = array;
 }

 @Override
 public double get(int index) {
  return array[index];
 }

 @Override
 public int getInt(int index) {
  return (int)array[index];
 }

 @Override
 public int size() {
  return array.length;
 }

 @Override
 public Iterator<Double> iterator(){
  return new Iterator<Double>() {
   private int i = 0;
   @Override
   public boolean hasNext() {
    return i < array.length;
   }

   @Override
   public Double next() {
    return array[i++];
   }
   
  };
 }

 @Override
 public void add(int index, double value) {
  array[index] += value;
 }

 @Override
 public void add(int index, int value) {
  array[index] += value;
 }

 @Override
 public Vector sigmoid() {
  Util.sigmoid(array);
  return this;
 }

 @Override
 public Vector softmax() {
  Util.softmax(array);
  return this;
 }

 @Override
 public Vector binomial(Random rng){
  return create(Util.binomial(rng, array));
 }

 @Override
 public Vector copy(){
  return new VectorDense(copyArray());
 }

 @Override
 public double[] copyArray(){
  return Arrays.copyOf(array, array.length);
 }

}
