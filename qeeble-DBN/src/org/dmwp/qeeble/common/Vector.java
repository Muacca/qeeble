package org.dmwp.qeeble.common;

import java.util.Random;

public interface Vector extends Iterable<Double>{

 public double get(int index);
 public int getInt(int index);
 public void add(int index, double value);
 public void add(int index, int value);
 public int size();
 
 public Vector sigmoid();
 public Vector softmax();
 public Vector binomial(Random rng);

 public Vector copy();
 public double[] copyArray();
}
