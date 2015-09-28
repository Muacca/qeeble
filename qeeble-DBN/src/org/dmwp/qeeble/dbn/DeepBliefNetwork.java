package org.dmwp.qeeble.dbn;

import java.util.List;
import java.util.Random;

import org.dmwp.qeeble.Model;
import org.dmwp.qeeble.Util;
import org.dmwp.qeeble.lr.LogisticRegression;
import org.dmwp.qeeble.rbm.RestrictedBoltzmannMachine;

public class DeepBliefNetwork {

 public static void pretrain(Random rng, Model model, List<Model> preModels, int[] x, double learningRate, int k) throws Exception {
  int[] layer_input = x;
  for(Model preModel: preModels) {
   layer_input = Util.binomial(rng, preModel.visible2Hidden(layer_input));
  }
  RestrictedBoltzmannMachine.train(model, rng, layer_input, learningRate, k);
 }

 public static void finetune(Random rng, Model outputModel, List<Model> models, int[] x, int[] y, double lr) throws Exception {
  int[] layer_input = x;
  for(Model model: models) {
   layer_input = Util.binomial(rng, model.visible2Hidden(layer_input));
  }
  LogisticRegression.train(outputModel, layer_input, y, lr);
 }

 public static double[] predict(Model outputModel, List<Model> models, int[] x) throws Exception {
  double[] layer_input = new double[x.length];
  for(int j = 0; j < x.length; j++) {
   layer_input[j] = x[j];
  }
  for(Model model: models) {
   layer_input = model.visible2Hidden(layer_input);
  }
  return LogisticRegression.predict(outputModel, layer_input);
 }

 private DeepBliefNetwork() {}

}
