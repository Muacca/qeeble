package org.dmwp.qeeble.dbn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.dmwp.qeeble.ColumnInfo;
import org.dmwp.qeeble.ColumnInfoDefault;
import org.dmwp.qeeble.ColumnInfoIndexed;
import org.dmwp.qeeble.DumpUtil;
import org.dmwp.qeeble.Model;
import org.dmwp.qeeble.lr.LogisticRegressionContext;
import org.dmwp.qeeble.rbm.RestrictedBoltzmannMachineContext;

public class DeepBeliefNetworkTest {

 // training data
 private static int[][] train_X = { //TODO 今は先頭に説明変数がないとだめ。目的変数は末尾に配置。
  {1, 1, 1, 0, 0, 0, 1, 0},
  {1, 0, 1, 0, 0, 0, 1, 0},
  {1, 1, 1, 0, 0, 0, 1, 0},
  {0, 0, 1, 1, 1, 0, 0, 1},
  {0, 0, 1, 1, 0, 0, 0, 1},
  {0, 0, 1, 1, 1, 0, 0, 1}
 };

 // test data
 private static int[][] test_X = {
  {1, 1, 0, 0, 0, 0, 0, 0},
  {1, 1, 1, 1, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 0, 0, 0},
  {0, 0, 1, 1, 1, 0, 0, 0},
 };

 // params
 private static final double pretrain_lr = 0.1 / train_X.length; //学習レコード数で割ると良いらしい。
 private static final int pretraining_epochs = 1000;
 private static final int cdSteps = 1;
 private static final double finetune_lr = 0.1 / train_X.length; //学習レコード数で割ると良いらしい。
 private static final int finetune_epochs = 500;

 private static final ColumnInfo outputLayer = new ColumnInfoIndexed(new int[]{6, 7});
 private static final ColumnInfo[] RBMlayers = {
  new ColumnInfoIndexed(new int[] {0, 1, 2, 3, 4, 5}),
  new ColumnInfoDefault(3),
  new ColumnInfoDefault(3)
 };

 public static void main(String[] args) throws Exception {
  DeepBeliefNetworkContext context = new DeepBeliefNetworkContext(
   new RestrictedBoltzmannMachineContext(new Random(1234), pretrain_lr, cdSteps),
   new LogisticRegressionContext(finetune_lr));

  // pretrain(construct multi-layer RBM, Deep Blief Network)
  List<Model> models = new ArrayList<Model>();
  for(int i = 0; i < RBMlayers.length - 1; ++i) {
   Model model = context.getPreContext().create(RBMlayers[i], RBMlayers[i + 1]);
   for(int epoch = 0; epoch < pretraining_epochs; ++epoch) {
    for(int[] x: train_X) {
     DeepBeliefNetwork.pretrain(context, model, models, x);
    }
   }
   models.add(model);
  }

  // finetune(Logistic Regression)
  Model outputModel = Model.createEmpty(RBMlayers[RBMlayers.length - 1], outputLayer);
  for(int epoch = 0; epoch < finetune_epochs; ++epoch) {
   for(int[] x: train_X) {
    DeepBeliefNetwork.finetune(context, outputModel, models, x);
   }
  }

  // predict
  for(int[] x: test_X) {
   DumpUtil.dumpArray(DeepBeliefNetwork.predict(outputModel, models, x));
  }
 }

 private DeepBeliefNetworkTest(){}
}
