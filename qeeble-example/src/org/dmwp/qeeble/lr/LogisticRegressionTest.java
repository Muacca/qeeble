package org.dmwp.qeeble.lr;

import org.dmwp.qeeble.DumpUtil;
import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.VectorDense;

public class LogisticRegressionTest {

 private static int visibleColumnSize = 6;
 private static int hiddenColumnSize = 2;
 
 // training data: index=0:input index=1:output
 private static int[][][] train_XY = {
  {{1, 1, 1, 0, 0, 0}, {1, 0}},
  {{1, 0, 1, 0, 0, 0}, {1, 0}},
  {{1, 1, 1, 0, 0, 0}, {1, 0}},
  {{0, 0, 1, 1, 1, 0}, {0, 1}},
  {{0, 0, 1, 1, 0, 0}, {0, 1}},
  {{0, 0, 1, 1, 1, 0}, {0, 1}}
 };
  
 // test data
 private static int[][] test_X = {
  {1, 0, 1, 0, 0, 0},
  {0, 0, 1, 1, 1, 0}
 };
 
 private static final double learningRate = 0.1;
 private static final int n_epochs = 500;
 
 public static void main(String[] args) throws Exception {
  LogisticRegressionContext context = new LogisticRegressionContext(learningRate, n_epochs);
  Model model = context.create(visibleColumnSize, hiddenColumnSize);
  
  // train 
  while(context.hasNext()) {
   for(int[][] xy: train_XY) {
    LogisticRegression.train(context, model, VectorDense.create(xy[0]), VectorDense.create(xy[1]));
   }
   context.next();
  }
  
  // predict
  for(int[] x: test_X) {
   DumpUtil.dump(LogisticRegression.predict(model, VectorDense.create(x)));
  }
 }

 private LogisticRegressionTest() {}
}
