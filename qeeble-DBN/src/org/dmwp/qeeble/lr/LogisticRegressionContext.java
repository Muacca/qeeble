package org.dmwp.qeeble.lr;

import org.dmwp.qeeble.ColumnInfo;
import org.dmwp.qeeble.Model;

public class LogisticRegressionContext {

 private double learningRate;
 public LogisticRegressionContext(double learningRate) {
  super();
  this.learningRate = learningRate;
 }
 
 public Model create(ColumnInfo visibleColumns, ColumnInfo hiddenColumns) {
  return Model.createEmpty(visibleColumns, hiddenColumns);
 }
 
 public double getLearningRate() {
  return learningRate;
 }
 public void setLearningRate(double learningRate) {
  this.learningRate = learningRate;
 }

}
