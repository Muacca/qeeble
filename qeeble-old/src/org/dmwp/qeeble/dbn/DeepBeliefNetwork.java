package org.dmwp.qeeble.dbn;

import java.util.List;

import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.Vector;
import org.dmwp.qeeble.lr.LogisticRegression;
import org.dmwp.qeeble.rbm.RestrictedBoltzmannMachine;

public class DeepBeliefNetwork {

 public static void pretrain(DeepBeliefNetworkContext context, Model model, List<Model> preModels, Vector x) throws Exception {
  Vector layer_input = x;
  for(Model preModel: preModels) {
   layer_input = preModel.visible2Hidden(layer_input);
  }
  RestrictedBoltzmannMachine.train(context.getPreContext(), model, layer_input);
 }

 public static void pretrain(DeepBeliefNetworkContext context, Model model, List<Model> preModels, double[] x) throws Exception {
  double[] layer_input = x;
  for(Model preModel: preModels) {
   layer_input = preModel.visible2Hidden(layer_input);
  }
  RestrictedBoltzmannMachine.train(context.getPreContext(), model, layer_input);
 }

 public static void pretrain(DeepBeliefNetworkContext context, Model model, List<Model> preModels, int[] x) throws Exception {
  int[] layer_input = x;
  for(Model preModel: preModels) {
   layer_input = context.getPreContext().binomial(preModel.visible2Hidden(layer_input));
  }
  RestrictedBoltzmannMachine.train(context.getPreContext(), model, layer_input);
 }

 public static void finetune(DeepBeliefNetworkContext context, Model outputModel, List<Model> models, Vector x, Vector y) throws Exception {
  Vector layer_input = x;
  for(Model model: models) {
   layer_input = model.visible2Hidden(layer_input);
  }
  LogisticRegression.train(context.getFineContext(), outputModel, layer_input, y);
 }

 public static void finetune(DeepBeliefNetworkContext context, Model outputModel, List<Model> models, double[] x, double[] y) throws Exception {
  double[] layer_input = x;
  for(Model model: models) {
   layer_input = model.visible2Hidden(layer_input);
  }
  LogisticRegression.train(context.getFineContext(), outputModel, layer_input, y);
 }

 public static void finetune(DeepBeliefNetworkContext context, Model outputModel, List<Model> models, int[] x, int[] y) throws Exception {
  int[] layer_input = x;
  for(Model model: models) {
   layer_input = context.getPreContext().binomial(model.visible2Hidden(layer_input));
  }
  LogisticRegression.train(context.getFineContext(), outputModel, layer_input, y);
 }

 public static Vector predict(Model outputModel, List<Model> models, Vector x) throws Exception {
  Vector layer_input = x;
  for(Model model: models) {
   layer_input = model.visible2Hidden(layer_input);
  }
  return LogisticRegression.predict(outputModel, layer_input);
 }

 public static double[] predict(Model outputModel, List<Model> models, double[] x) throws Exception {
  double[] layer_input = x;
  for(Model model: models) {
   layer_input = model.visible2Hidden(layer_input);
  }
  return LogisticRegression.predict(outputModel, layer_input);
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
 
 private DeepBeliefNetwork() {}

}
