package org.dmwp.qeeble.rbm;

import java.util.Random;

import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.Util;
import org.dmwp.qeeble.common.Vector;

public class RestrictedBoltzmannMachineContext {

 public static final int DEFAULT_CONTRASTIVE_DIVERGENCE_STEP = 1;

 private Random rng;
 private double learningRate;
 private int contrastiveDivergenceStep;
 public RestrictedBoltzmannMachineContext(Random rng, double learningRate) {
  this(rng, learningRate, DEFAULT_CONTRASTIVE_DIVERGENCE_STEP);
 }
 public RestrictedBoltzmannMachineContext(Random rng, double learningRate, int contrastiveDivergenceStep) {
  super();
  this.rng = rng;
  this.learningRate = learningRate;
  this.contrastiveDivergenceStep = contrastiveDivergenceStep;
 }

 public Model create(int visibleColumnSize, int hiddenColumnSize) {
  return Model.createUniformed(rng, visibleColumnSize, hiddenColumnSize);
 }

 public int[] binomial(double[] array) {
  return Util.binomial(rng, array);
 }

 public Vector binomial(Vector v) {
  return v.binomial(rng);
 }


 public double getLearningRate() {
  return learningRate;
 }
 public int getContrastiveDivergenceStep() {
  return contrastiveDivergenceStep;
 }


}
