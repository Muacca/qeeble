package org.dmwp.qeeble.rbm;

import org.dmwp.qeeble.common.MatrixCrossVector;
import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.Vector;

public class RestrictedBoltzmannMachine {

 private static void tune(Model model, MatrixCrossVector m1, MatrixCrossVector m2, double v) {
  for(int i = 0; i < m1.rowSize(); ++i) {
   for(int j = 0; j < m1.columnSize(); ++j) {
    model.addWeight(i, j, (m1.get(i, j) - m2.get(i, j)) * v);
   }
  }
 }
 
 public static void train(RestrictedBoltzmannMachineContext context, Model model, Vector input) throws Exception {
  Vector ph_mean = model.visible2Hidden(input);
  Vector ph_sample = context.binomial(ph_mean);
  Vector nv_sample = context.binomial(model.hidden2Visible(ph_sample));
  Vector nh_mean = model.visible2Hidden(nv_sample);
  
  tune(model, MatrixCrossVector.create(ph_mean, input), MatrixCrossVector.create(nh_mean, nv_sample), context.getLearningRate());
  
  ph_sample.subtract(nh_mean);
  ph_sample.multiply(context.getLearningRate());
  model.addHiddenBias(ph_sample);

  nv_sample.subtractFrom(input);
  nv_sample.multiply(context.getLearningRate());
  model.addVisibleBias(nv_sample);
 }

 public static Vector reconstruct(Model model, Vector input) throws Exception {
  return model.hidden2Visible(model.visible2Hidden(input));
 }

 private RestrictedBoltzmannMachine() {}

}
