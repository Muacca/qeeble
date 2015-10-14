package org.dmwp.qeeble.common;

import java.util.Random;

public interface Vector extends Iterable<Double>{

 public int size();
 public double get(int index);
 public int getInt(int index);

 public void add(int index, double value);
 public void add(int index, int value);
 public void add(Vector v) throws Exception;
 public void subtract(int index, double value);
 public void subtract(int index, int value);
 public void subtract(Vector v) throws Exception;
 public void subtractFrom(int index, double value);
 public void subtractFrom(int index, int value);
 public void subtractFrom(Vector v) throws Exception;
 public void multiply(double value);
 
 public Vector sigmoid();
 public Vector softmax();
 public Vector binomial(Random rng);

 public Vector copy();
 public double[] copyToArray();
}
