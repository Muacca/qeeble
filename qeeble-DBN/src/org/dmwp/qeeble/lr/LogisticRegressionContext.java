package org.dmwp.qeeble.lr;

import org.dmwp.qeeble.common.Model;

public class LogisticRegressionContext {

 private double learningRate;
 public LogisticRegressionContext(double learningRate) {
  super();
  this.learningRate = learningRate;
 }
 
 public Model create(int visibleColumnSize, int hiddenColumnSize) {
  return Model.createEmpty(visibleColumnSize, hiddenColumnSize);
 }
 
 public double getLearningRate() {
  return learningRate;
 }

}
