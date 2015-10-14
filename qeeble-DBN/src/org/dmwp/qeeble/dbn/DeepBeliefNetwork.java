package org.dmwp.qeeble.dbn;

import java.util.List;

import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.Vector;
import org.dmwp.qeeble.lr.LogisticRegression;
import org.dmwp.qeeble.rbm.RestrictedBoltzmannMachine;

public class DeepBeliefNetwork {

 public static void pretrain(DeepBeliefNetworkContext context, Model model, List<Model> preModels, Vector x) throws Exception {
  RestrictedBoltzmannMachine.train(context.getPreContext(), model, forwarding(preModels, x));
 }

 public static void finetune(DeepBeliefNetworkContext context, Model outputModel, List<Model> preModels, Vector x, Vector y) throws Exception {
  LogisticRegression.train(context.getFineContext(), outputModel, forwarding(preModels, x), y);
 }

 public static Vector predict(Model outputModel, List<Model> preModels, Vector x) throws Exception {
  return LogisticRegression.predict(outputModel, forwarding(preModels, x));
 }

 public static Vector forwarding(List<Model> preModels, Vector x) throws Exception {
  for(Model model: preModels) {
   x = model.visible2Hidden(x);
  }
  return x;
 }
 
 private DeepBeliefNetwork() {}

}
