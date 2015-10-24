package org.dmwp.qeeble.common;

import java.util.Iterator;
import java.util.Random;

public class VectorIndexed implements Vector {
 
 public static VectorIndexed create(int[] indexs, double[] array) {
  return new VectorIndexed(indexs, array);
 }
 
 private int[] indexs;
 private double[] array;
 private VectorIndexed(int[] indexs, double[] array){
  super();
  this.indexs = indexs;
  this.array = array;
 }

 @Override
 public final double get(int index) {
  return array[indexs[index]];
 }

 @Override
 public final int getInt(int index) {
  return (int)array[indexs[index]];
 }

 @Override
 public final int size() {
  return indexs.length;
 }

 @Override
 public Iterator<Double> iterator(){
  return new Iterator<Double>() {
   private int i = 0;
   @Override
   public boolean hasNext() {
    return i < indexs.length;
   }
   @Override
   public Double next() {
    return array[indexs[i++]];
   }
  };
 }

 @Override
 public final void add(int index, double value) {
  array[indexs[index]] += value;
 }

 @Override
 public final void add(int index, int value) {
  array[indexs[index]] += value;
 }

 @Override
 public void add(Vector v) throws Exception {
  if(v.size() != size()) throw new Exception("invalid dimension.");
  for(int i = 0; i < indexs.length; ++i) {
   add(i, v.get(i));
  }
 }

 @Override
 public final void subtract(int index, double value) {
  array[indexs[index]] -= value;
 }

 @Override
 public final void subtract(int index, int value) {
  array[indexs[index]] -= value;
 }

 @Override
 public final void subtractFrom(int index, double value) {
  array[indexs[index]] = value - array[indexs[index]];
 }

 @Override
 public final void subtractFrom(int index, int value) {
  array[indexs[index]] = value - array[indexs[index]];
 }

 @Override
 public void subtract(Vector v) throws Exception {
  if(v.size() != size()) throw new Exception("invalid dimension.");
  for(int i = 0; i < indexs.length; ++i) {
   subtract(i, v.get(i));
  }
 }

 @Override
 public void subtractFrom(Vector v) throws Exception {
  if(v.size() != size()) throw new Exception("invalid dimension:" + v.size() + ", must:" + size());
  for(int i = 0; i < indexs.length; ++i) {
   subtractFrom(i, v.get(i));
  }
 }

 @Override
 public final void multiply(double value) {
  for(int i = 0; i < indexs.length; ++i) {
   array[indexs[i]] *= value;
  }
 }

 @Override
 public Vector sigmoid() {
  return VectorDense.create(Util.sigmoid(copyToArray()));
 }

 @Override
 public Vector softmax() {
  return VectorDense.create(Util.softmax(copyToArray()));
 }

 @Override
 public Vector binomial(Random rng){
  return VectorDense.create(Util.binomial(rng, copyToArray()));
 }

 @Override
 public Vector copy(){
  return VectorDense.create(copyToArray());
 }

 @Override
 public double[] copyToArray(){
  double[] temp = new double[indexs.length];
  for(int i = 0; i < indexs.length; ++i) {
   temp[i] = array[indexs[i]];
  }
  return temp;
 }

}
