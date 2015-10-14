package org.dmwp.qeeble.rbm;

import java.util.Random;

import org.dmwp.qeeble.common.EpochContext;
import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.Vector;

public class RestrictedBoltzmannMachineContext extends EpochContext {

 private Random rng;
 private double learningRate0;
 private double learningRate;
 public RestrictedBoltzmannMachineContext(Random rng, double learningRate, int epochMax) {
  super(epochMax);
  this.rng = rng;
  this.learningRate0 = learningRate;
  this.learningRate = learningRate;
 }

 public Model create(int visibleColumnSize, int hiddenColumnSize) {
  return Model.createUniformed(rng, visibleColumnSize, hiddenColumnSize);
 }

 public Vector binomial(Vector v) {
  return v.binomial(rng);
 }


 public double getLearningRate() {
  return learningRate;
 }

 @Override
 public void next() {
  super.next();
  learningRate = learningRate0 / Math.pow(1 + learningRate0 * currentEpoch(), 0.75);
 }

 @Override
 public void initEpoch() {
  super.initEpoch();
  learningRate = learningRate0;
 }

}
