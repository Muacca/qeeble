package org.dmwp.qeeble.rbm;

import java.util.Random;

import org.dmwp.qeeble.ColumnInfo;
import org.dmwp.qeeble.ColumnInfoDefault;
import org.dmwp.qeeble.ColumnInfoIndexed;
import org.dmwp.qeeble.DumpUtil;
import org.dmwp.qeeble.Model;

public class RestrictedBoltzmannMachineTestDouble {

 private static ColumnInfo visibleColumns = new ColumnInfoIndexed(new int[]{
  0, 1, 2, 3, 4, 5
 });
 private static ColumnInfo hiddenColumns = new ColumnInfoDefault(3);

 // training data
 private static double[][] train_X = {
  {1, 1, 1, 0, 0, 0},
  {1, 0, 1, 0, 0, 0},
  {1, 1, 1, 0, 0, 0},
  {0, 0, 1, 1, 1, 0},
  {0, 0, 1, 0, 1, 0},
  {0, 0, 1, 1, 1, 0}
 };
 
 
 // test data
 private static double[][] test_X = {
  {1, 1, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 0}
 };
 
 private static final double learningRate = 0.1 / train_X.length; //学習レコード数で割ると良いらしい。
 private static final int training_epochs = 1000;
 private static final int cdSteps = 1;

 public static void main(String[] args) throws Exception {
  RestrictedBoltzmannMachineContext context = new RestrictedBoltzmannMachineContext(
   new Random(1234), learningRate, cdSteps
  );
  Model model = context.create(visibleColumns, hiddenColumns);

  // train
  for(int epoch = 0; epoch < training_epochs; epoch++) {
   for(double[] x: train_X) {
    RestrictedBoltzmannMachine.train(context, model, x);
   }
  }

  // reconstruct
  for(double[] x: test_X) {
   DumpUtil.dumpArray(RestrictedBoltzmannMachine.reconstruct(model, x));
  }
 }

 private RestrictedBoltzmannMachineTestDouble() {}
}
