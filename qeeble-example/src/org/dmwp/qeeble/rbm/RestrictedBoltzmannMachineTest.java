package org.dmwp.qeeble.rbm;

import java.util.Random;

import org.dmwp.qeeble.DumpUtil;
import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.VectorDense;

public class RestrictedBoltzmannMachineTest {

 private static int visibleColumnSize = 6;
 private static int hiddenColumnSize = 3;

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
 
 private static final double learningRate = 0.1;
 private static final int training_epochs = 1000;

 public static void main(String[] args) throws Exception {
  RestrictedBoltzmannMachineContext context = new RestrictedBoltzmannMachineContext(new Random(1234), learningRate, training_epochs);
  Model model = context.create(visibleColumnSize, hiddenColumnSize);

  // train
  while(context.hasNext()) {
   for(double[] x: train_X) {
    RestrictedBoltzmannMachine.train(context, model, VectorDense.create(x));
   }
   context.next();
  }

  // reconstruct
  for(double[] x: test_X) {
   DumpUtil.dump(RestrictedBoltzmannMachine.reconstruct(model, VectorDense.create(x)));
  }
 }

 private RestrictedBoltzmannMachineTest() {}
}
