package org.dmwp.qeeble.lr;

import org.dmwp.qeeble.Model;
import org.dmwp.qeeble.Util;

public class LogisticRegression {
 
 public static void train(LogisticRegressionContext context, Model model, double[] input, double[] output) throws Exception {
  double[] py = predict(model, input);
  for(int i = 0; i < model.getHiddenColumnInfo().size(); i++) {
   double dy = (output[model.getHiddenColumnInfo().index(i)] - py[i]) * context.getLearningRate();
   for(int j = 0; j < model.getVisibleColumnInfo().size(); j++) {
    model.addWeight(i, j, dy * input[model.getVisibleColumnInfo().index(j)]);
   }
   model.addHiddenBias(i, dy);
  }
 }
 
 public static void train(LogisticRegressionContext context, Model model, int[] input, int[] output) throws Exception {
  double[] py = predict(model, input);
  for(int i = 0; i < model.getHiddenColumnInfo().size(); i++) {
   double dy = (output[model.getHiddenColumnInfo().index(i)] - py[i]) * context.getLearningRate();
   for(int j = 0; j < model.getVisibleColumnInfo().size(); j++) {
    model.addWeight(i, j, dy * input[model.getVisibleColumnInfo().index(j)]);
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

 private LogisticRegression() {}
 
}
