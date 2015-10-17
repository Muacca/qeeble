package org.dmwp.qeeble.dbn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.VectorDense;
import org.dmwp.qeeble.lr.LogisticRegressionContext;
import org.dmwp.qeeble.rbm.RestrictedBoltzmannMachineContext;
import org.dmwp.qeeble.util.DumpUtil;

public class DeepBeliefNetworkTestVector {

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
 private static final double pretrain_lr = 0.1 / train_XY.length; //学習レコード数で割ると良いらしい。
 private static final int pretraining_epochs = 1000;
 private static final int cdSteps = 1;
 private static final double finetune_lr = 0.1 / train_XY.length; //学習レコード数で割ると良いらしい。
 private static final int finetune_epochs = 500;

 private static final int outputLayerSize = 2;
 private static final int[] RBMlayerSizes = {6, 3, 3};

 public static void main(String[] args) throws Exception {
  DeepBeliefNetworkContext context = new DeepBeliefNetworkContext(
   new RestrictedBoltzmannMachineContext(new Random(1234), pretrain_lr, cdSteps),
   new LogisticRegressionContext(finetune_lr));

  // pretrain(construct multi-layer RBM, Deep Blief Network)
  List<Model> models = new ArrayList<Model>();
  for(int i = 0; i < RBMlayerSizes.length - 1; ++i) {
   Model model = context.getPreContext().create(RBMlayerSizes[i], RBMlayerSizes[i + 1]);
   for(int epoch = 0; epoch < pretraining_epochs; ++epoch) {
    for(double[][] xy: train_XY) {
     DeepBeliefNetwork.pretrain(context, model, models, VectorDense.create(xy[0]));
    }
   }
   models.add(model);
  }

  // finetune(Logistic Regression)
  Model outputModel = Model.createEmpty(RBMlayerSizes[RBMlayerSizes.length - 1], outputLayerSize);
  for(int epoch = 0; epoch < finetune_epochs; ++epoch) {
   for(double[][] xy: train_XY) {
    DeepBeliefNetwork.finetune(context, outputModel, models, VectorDense.create(xy[0]), VectorDense.create(xy[1]));
   }
  }

  // predict
  for(double[] x: test_X) {
   DumpUtil.dump(DeepBeliefNetwork.predict(outputModel, models, VectorDense.create(x)));
  }
 }

 private DeepBeliefNetworkTestVector(){}
}
