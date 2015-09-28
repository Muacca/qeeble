package org.dmwp.qeeble.lr;

import org.dmwp.qeeble.ColumnInfo;
import org.dmwp.qeeble.DumpUtil;
import org.dmwp.qeeble.Model;

public class LogisticRegressionTest {

 private static final int[] visibleColumns = {
  0, 1, 2, 3, 4, 5
 };
 
 private static final int[] hiddenColumns = {
  0, 1
 };

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
  {0, 1}
 }; 
 
 // test data
 private static int[][] test_X = {
  {1, 0, 1, 0, 0, 0},
  {0, 0, 1, 1, 1, 0}
 };
 
 private static final double learningRate = 0.1 / train_X.length; //学習レコード数で割ると良いらしい。
 private static final int n_epochs = 500;
 
 public static void main(String[] args) throws Exception {
  LogisticRegressionContext context = new LogisticRegressionContext(learningRate, n_epochs);
  Model model = context.create(ColumnInfo.create(visibleColumns), ColumnInfo.create(hiddenColumns));
  
  // train 
  for(int epoch = 0; epoch < context.getEpochCount(); epoch++) {
   for(int i = 0; i < train_X.length; i++) { //TODO 学習データと正答データを１レコードにする。
    LogisticRegression.train(model, train_X[i], train_Y[i], context.getLearningRate());
   }
  }
  
  // predict
  for(int[] x: test_X) {
   DumpUtil.dumpArray(LogisticRegression.predict(model, x));
  }
 }

 private LogisticRegressionTest() {}
}
