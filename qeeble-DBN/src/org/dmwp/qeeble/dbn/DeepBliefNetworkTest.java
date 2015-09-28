package org.dmwp.qeeble.dbn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.dmwp.qeeble.ColumnInfo;
import org.dmwp.qeeble.DumpUtil;
import org.dmwp.qeeble.Model;
import org.dmwp.qeeble.lr.LogisticRegressionContext;
import org.dmwp.qeeble.rbm.RestrictedBoltzmannMachineContext;

public class DeepBliefNetworkTest {

 // training data
 private static int[][] train_X = {
  {1, 1, 1, 0, 0, 0},
  {1, 0, 1, 0, 0, 0},
  {1, 1, 1, 0, 0, 0},
  {0, 0, 1, 1, 1, 0},
  {0, 0, 1, 1, 0, 0},
  {0, 0, 1, 1, 1, 0}
 };

 private static int[][] train_Y = {
  {1, 0},
  {1, 0},
  {1, 0},
  {0, 1},
  {0, 1},
  {0, 1},
 };

 // test data
 private static int[][] test_X = {
  {1, 1, 0, 0, 0, 0},
  {1, 1, 1, 1, 0, 0},
  {0, 0, 0, 1, 1, 0},
  {0, 0, 1, 1, 1, 0},
 };

 // params
 private static final double pretrain_lr = 0.1 / train_X.length; //学習レコード数で割ると良いらしい。
 private static final int pretraining_epochs = 1000;
 private static final int cdSteps = 1;
 private static final double finetune_lr = 0.1 / train_X.length; //学習レコード数で割ると良いらしい。
 private static final int finetune_epochs = 500;

 private static final ColumnInfo outputLayer = ColumnInfo.create(new int[]{0, 1});
 private static final ColumnInfo[] RBMlayers = {
  ColumnInfo.create(new int[] {0, 1, 2, 3, 4, 5}),
  ColumnInfo.create(new int[] {0, 1, 2}),
  ColumnInfo.create(new int[] {0, 1, 2})
 };

 public static void main(String[] args) throws Exception {

  // pretrain(construct multi-layer RBM, Deep Blief Network)
  RestrictedBoltzmannMachineContext preContext = new RestrictedBoltzmannMachineContext(
   new Random(1234), pretrain_lr, pretraining_epochs, cdSteps);
  List<Model> models = new ArrayList<Model>();
  for(int i = 0; i < RBMlayers.length - 1; ++i) {
   Model model = preContext.create(RBMlayers[i], RBMlayers[i + 1]);
   for(int epoch = 0; epoch < preContext.getEpochCount(); epoch++) {
    for(int[] x: train_X) {
     DeepBliefNetwork.pretrain(preContext.getRng(), model, models, x,
      preContext.getLearningRate(), preContext.getContrastiveDivergenceStep());
    }
   }
   models.add(model);
  }

  // finetune(Logistic Regression)
  LogisticRegressionContext fineContext = new LogisticRegressionContext(finetune_lr, finetune_epochs);
  Model outputModel = Model.createEmpty(RBMlayers[RBMlayers.length - 1], outputLayer);
  for(int epoch = 0; epoch < fineContext.getEpochCount(); epoch++) {
   for(int n = 0; n < train_X.length; n++) {
    DeepBliefNetwork.finetune(preContext.getRng(), outputModel, models, train_X[n], train_Y[n], fineContext.getLearningRate());
   }
  }

  // predict
  for(int[] x: test_X) {
   DumpUtil.dumpArray(DeepBliefNetwork.predict(outputModel, models, x));
  }
 }

 private DeepBliefNetworkTest(){}
}
