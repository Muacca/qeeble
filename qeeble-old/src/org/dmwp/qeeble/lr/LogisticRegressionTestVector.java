package org.dmwp.qeeble.lr;

import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.VectorDense;
import org.dmwp.qeeble.util.DumpUtil;

public class LogisticRegressionTestVector {

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
 
 private static final double learningRate = 0.1 / train_XY.length; //学習レコード数で割ると良いらしい。
 private static final int n_epochs = 500;
 
 public static void main(String[] args) throws Exception {
  LogisticRegressionContext context = new LogisticRegressionContext(learningRate);
  Model model = context.create(visibleColumnSize, hiddenColumnSize);
  
  // train 
  for(int epoch = 0; epoch < n_epochs; epoch++) {
   for(int[][] xy: train_XY) {
    LogisticRegression.train(context, model, VectorDense.create(xy[0]), VectorDense.create(xy[1]));
   }
  }
  
  // predict
  for(int[] x: test_X) {
   DumpUtil.dump(LogisticRegression.predict(model, VectorDense.create(x)));
  }
 }

 private LogisticRegressionTestVector() {}
}
