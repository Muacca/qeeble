package org.dmwp.qeeble.dbn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.dmwp.qeeble.DumpUtil;
import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.VectorDense;
import org.dmwp.qeeble.lr.LogisticRegressionContext;
import org.dmwp.qeeble.rbm.RestrictedBoltzmannMachineContext;

public class DeepBeliefNetworkTest {

 // training data: index=0:input index=1:output
 private static double[][][] train_XY = {
  {{1, 1, 1, 0, 0, 0}, {1, 0}},
  {{1, 0, 1, 0, 0, 0}, {1, 0}},
  {{1, 1, 1, 0, 0, 0}, {1, 0}},
  {{0, 0, 1, 1, 1, 0}, {0, 1}},
  {{0, 0, 1, 1, 0, 0}, {0, 1}},
  {{0, 0, 1, 1, 1, 0}, {0, 1}}
 };

 // test data
 private static double[][] test_X = {
  {1, 1, 0, 0, 0, 0},
  {1, 1, 1, 1, 0, 0},
  {0, 0, 0, 1, 1, 0},
  {0, 0, 1, 1, 1, 0},
 };

 // params
 private static final double pretrain_lr = 0.1;
 private static final int pretraining_epochs = 1000;
 private static final double finetune_lr = 0.1;
 private static final int finetune_epochs = 500;

 private static final int outputLayerSize = 2;
 private static final int[] RBMlayerSizes = {6, 3, 3};

 public static void main(String[] args) throws Exception {
  DeepBeliefNetworkContext context = new DeepBeliefNetworkContext(
   new RestrictedBoltzmannMachineContext(new Random(1234), pretrain_lr, pretraining_epochs),
   new LogisticRegressionContext(finetune_lr, finetune_epochs));

  // pretrain(construct multi-layer RBM, Deep Blief Network)
  List<Model> preModels = new ArrayList<Model>();
  for(int i = 0; i < RBMlayerSizes.length - 1; ++i) {
   Model model = context.getPreContext().create(RBMlayerSizes[i], RBMlayerSizes[i + 1]);
   while(context.getPreContext().hasNext()) {
    for(double[][] xy: train_XY) {
     DeepBeliefNetwork.pretrain(context, model, preModels, VectorDense.create(xy[0]));
    }
    context.getPreContext().next();
   }
   preModels.add(model);
   context.getPreContext().initEpoch();
  }
  
  // finetune(Logistic Regression)
  Model outputModel = Model.createEmpty(RBMlayerSizes[RBMlayerSizes.length - 1], outputLayerSize);
  while(context.getFineContext().hasNext()) {
   for(double[][] xy: train_XY) {
    DeepBeliefNetwork.finetune(context, outputModel, preModels, VectorDense.create(xy[0]), VectorDense.create(xy[1]));
   }
   context.getFineContext().next();
  }

  // predict
  for(double[] x: test_X) {
   DumpUtil.dump(DeepBeliefNetwork.predict(outputModel, preModels, VectorDense.create(x)));
  }
 }

 private DeepBeliefNetworkTest(){}
}
