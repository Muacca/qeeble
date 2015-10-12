package org.dmwp.qeeble.lr;

import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.Util;
import org.dmwp.qeeble.common.Vector;

public class LogisticRegression {
 
 public static void train(LogisticRegressionContext context, Model model, double[] input, double[] output) throws Exception {
  double[] py = predict(model, input);
  for(int i = 0; i < model.getHiddenColumnSize(); ++i) {
   double dy = (output[i] - py[i]) * context.getLearningRate();
   for(int j = 0; j < model.getVisibleColumnSize(); ++j) {
    model.addWeight(i, j, dy * input[j]);
   }
   model.addHiddenBias(i, dy);
  }
 }
 
 public static void train(LogisticRegressionContext context, Model model, int[] input, int[] output) throws Exception {
  double[] py = predict(model, input);
  for(int i = 0; i < model.getHiddenColumnSize(); ++i) {
   double dy = (output[i] - py[i]) * context.getLearningRate();
   for(int j = 0; j < model.getVisibleColumnSize(); ++j) {
    model.addWeight(i, j, dy * input[j]);
   }
   model.addHiddenBias(i, dy);
  }
 }
 
 public static void train(LogisticRegressionContext context, Model model, Vector input, Vector output) throws Exception {
  Vector py = predict(model, input);
  for(int i = 0; i < model.getHiddenColumnSize(); ++i) {
   double dy = (output.get(i) - py.get(i)) * context.getLearningRate();
   for(int j = 0; j < model.getVisibleColumnSize(); ++j) {
    model.addWeight(i, j, dy * input.get(j));
   }
   model.addHiddenBias(i, dy);
  }
 }
 

 public static double[] predict(Model model, int[] input) throws Exception {
  return Util.softmax(model.propagateUp(input));
 }

 public static double[] predict(Model model, double[] input) throws Exception {
  return Util.softmax(model.propagateUp(input));
 }

 public static Vector predict(Model model, Vector input) throws Exception {
  return model.propagateUp(input).softmax();
 }

 private LogisticRegression() {}
 
}
