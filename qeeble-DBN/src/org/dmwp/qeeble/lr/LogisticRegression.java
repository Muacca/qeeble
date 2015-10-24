package org.dmwp.qeeble.lr;

import org.dmwp.qeeble.common.MatrixCrossVector;
import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.Vector;

public class LogisticRegression {
  
 public static void train(LogisticRegressionContext context, Model model, Vector input, Vector output) throws Exception {
  Vector py = predict(model, input);
  py.subtractFrom(output);
  py.multiply(context.getLearningRate());
  model.addHiddenBias(py);
  model.addWeight(MatrixCrossVector.create(py, input));
 }
 
 public static Vector predict(Model model, Vector input) throws Exception {
  return model.propagateUp(input).softmax();
 }

 private LogisticRegression() {}
 
}
