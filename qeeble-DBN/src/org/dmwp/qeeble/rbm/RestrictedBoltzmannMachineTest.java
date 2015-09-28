package org.dmwp.qeeble.rbm;

import java.util.Random;

import org.dmwp.qeeble.ColumnInfo;
import org.dmwp.qeeble.DumpUtil;
import org.dmwp.qeeble.Model;

public class RestrictedBoltzmannMachineTest {

 private static final int[] visibleColumns = {
  0, 1, 2, 3, 4, 5
 };
 private static final int[] hiddenColumns = {
  0, 1, 2
 };

 // training data
 private static int[][] train_X = {
  {1, 1, 1, 0, 0, 0},
  {1, 0, 1, 0, 0, 0},
  {1, 1, 1, 0, 0, 0},
  {0, 0, 1, 1, 1, 0},
  {0, 0, 1, 0, 1, 0},
  {0, 0, 1, 1, 1, 0}
 };
 
 
 // test data
 private static int[][] test_X = {
  {1, 1, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 0}
 };
 
 private static final double learningRate = 0.1 / train_X.length; //学習レコード数で割ると良いらしい。
 private static final int training_epochs = 1000;
 private static final int cdSteps = 1;

 public static void main(String[] args) throws Exception {
  RestrictedBoltzmannMachineContext context = new RestrictedBoltzmannMachineContext(
   new Random(1234), learningRate, training_epochs, cdSteps
  );
  Model model = context.create(ColumnInfo.create(visibleColumns), ColumnInfo.create(hiddenColumns));

  // train
  for(int epoch = 0; epoch < context.getEpochCount(); epoch++) {
   for(int[] x: train_X) {
    RestrictedBoltzmannMachine.train(
     model, context.getRng(), x, context.getLearningRate(), context.getContrastiveDivergenceStep());
   }
  }

  // reconstruct
  for(int[] x: test_X) {
   DumpUtil.dumpArray(RestrictedBoltzmannMachine.reconstruct(model, x));
  }
 }

 private RestrictedBoltzmannMachineTest() {}
}