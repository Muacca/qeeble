package org.dmwp.qeeble.lr;

import org.dmwp.qeeble.ColumnInfo;
import org.dmwp.qeeble.Model;

public class LogisticRegressionContext {

 public static final int DEFAULT_EPOCH_COUNT = 500;
 
 private double learningRate;
 private int epochCount;
 public LogisticRegressionContext(double learningRate) {
  this(learningRate, DEFAULT_EPOCH_COUNT);
 }
 public LogisticRegressionContext(double learningRate, int epochCount) {
  super();
   this.learningRate = learningRate;
  this.epochCount = epochCount;
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
 public int getEpochCount() {
  return epochCount;
 }
 public void setEpochCount(int epochCount) {
  this.epochCount = epochCount;
 }

}
