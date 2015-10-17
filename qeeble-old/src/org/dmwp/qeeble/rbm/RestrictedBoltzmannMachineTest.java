package org.dmwp.qeeble.rbm;

import java.util.Random;

import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.util.DumpUtil;

public class RestrictedBoltzmannMachineTest {

 private static int visibleColumnSize = 6;
 private static int hiddenColumnSize = 3;

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
   new Random(1234), learningRate, cdSteps
   );
  Model model = context.create(visibleColumnSize, hiddenColumnSize);

  // train
  for(int epoch = 0; epoch < training_epochs; epoch++) {
   for(int[] x: train_X) {
    RestrictedBoltzmannMachine.train(context, model, x);
   }
  }

  // reconstruct
  for(int[] x: test_X) {
   DumpUtil.dump(RestrictedBoltzmannMachine.reconstruct(model, x));
  }
 }

 private RestrictedBoltzmannMachineTest() {}
}
