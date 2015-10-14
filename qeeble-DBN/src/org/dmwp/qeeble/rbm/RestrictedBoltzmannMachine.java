package org.dmwp.qeeble.rbm;

import org.dmwp.qeeble.common.Matrix;
import org.dmwp.qeeble.common.MatrixDense;
import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.Vector;

public class RestrictedBoltzmannMachine {

 public static void train(RestrictedBoltzmannMachineContext context, Model model, Vector input) throws Exception {
  Vector ph_mean = model.visible2Hidden(input);
  Vector ph_sample = context.binomial(ph_mean);
  Vector nv_sample = context.binomial(model.hidden2Visible(ph_sample));
  Vector nh_mean = model.visible2Hidden(nv_sample);
  
  Matrix dw = MatrixDense.cross(ph_mean, input);
  dw.subtract(MatrixDense.cross(nh_mean, nv_sample));
  dw.multiply(context.getLearningRate());
  model.addWeight(dw);
  
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
