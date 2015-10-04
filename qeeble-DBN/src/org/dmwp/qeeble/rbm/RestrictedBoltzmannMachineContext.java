package org.dmwp.qeeble.rbm;

import java.util.Random;

import org.dmwp.qeeble.ColumnInfo;
import org.dmwp.qeeble.Model;
import org.dmwp.qeeble.Util;

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
 
 public Model create(ColumnInfo visibleColumns, ColumnInfo hiddenColumns) {
  return Model.createUniformed(rng, visibleColumns, hiddenColumns);
 }

 public int[] binomial(double[] array) {
  return Util.binomial(rng, array);
 }
 
 
 public Random getRng() {
  return rng;
 }
 public void setRng(Random rng) {
  this.rng = rng;
 }
 public double getLearningRate() {
  return learningRate;
 }
 public void setLearningRate(double learningRate) {
  this.learningRate = learningRate;
 }
 public int getContrastiveDivergenceStep() {
  return contrastiveDivergenceStep;
 }
 public void setContrastiveDivergenceStep(int contrastiveDivergenceStep) {
  this.contrastiveDivergenceStep = contrastiveDivergenceStep;
 }

 
}
