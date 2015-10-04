package org.dmwp.qeeble.lr;

import org.dmwp.qeeble.ColumnInfo;
import org.dmwp.qeeble.ColumnInfoIndexed;
import org.dmwp.qeeble.DumpUtil;
import org.dmwp.qeeble.Model;

public class LogisticRegressionTest {

 private static ColumnInfo visibleColumns = new ColumnInfoIndexed(new int[]{
  2, 3, 4, 5, 6, 7
 });
 
 private static ColumnInfo hiddenColumns = new ColumnInfoIndexed(new int[]{
  0, 1
 });
 
 private static int[][] train_X = {
  {1, 0, 1, 1, 1, 0, 0, 0},
  {1, 0, 1, 0, 1, 0, 0, 0},
  {1, 0, 1, 1, 1, 0, 0, 0},
  {0, 1, 0, 0, 1, 1, 1, 0},
  {0, 1, 0, 0, 1, 1, 0, 0},
  {0, 1, 0, 0, 1, 1, 1, 0}
 };
  
 // test data
 private static int[][] test_X = {
  {0, 0, 1, 0, 1, 0, 0, 0},
  {0, 0, 0, 0, 1, 1, 1, 0}
 };
 
 private static final double learningRate = 0.1 / train_X.length; //学習レコード数で割ると良いらしい。
 private static final int n_epochs = 500;
 
 public static void main(String[] args) throws Exception {
  LogisticRegressionContext context = new LogisticRegressionContext(learningRate);
  Model model = context.create(visibleColumns, hiddenColumns);
  
  // train 
  for(int epoch = 0; epoch < n_epochs; epoch++) {
   for(int[] x: train_X) {
    LogisticRegression.train(context, model, x, x);
   }
  }
  
  // predict
  for(int[] x: test_X) {
   DumpUtil.dumpArray(LogisticRegression.predict(model, x));
  }
 }

 private LogisticRegressionTest() {}
}
