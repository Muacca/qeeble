package org.dmwp.qeeble;

import java.util.Arrays;
import java.util.Random;

public class Model {
 
 /** uniformedされた重み行列を持つモデルを作成する。
  * 
  * @param rng
  * @param columnInfo
  * @return
  */
 public static Model createUniformed(Random rng, ColumnInfo visibleColumns, ColumnInfo hiddenColumns) {
  return new Model(visibleColumns, hiddenColumns, Util.uniformedWeight(
   rng, visibleColumns.size(), hiddenColumns.size()));
 }
 
 /** ゼロ行列を重みとして持つモデルを作成する。
  * 
  * @param columnInfo
  * @return
  */
 public static Model createEmpty(ColumnInfo visibleColumns, ColumnInfo hiddenColumns) {
  return new Model(visibleColumns, hiddenColumns, Util.zeroWeight(
   visibleColumns.size(), hiddenColumns.size()));
 }

 private ColumnInfo visibleColumns;
 private ColumnInfo hiddenColumns;
 private double[][] weight;
 private double[] visibleBias;
 private double[] hiddenBias;
 private Model(ColumnInfo visibleColumns, ColumnInfo hiddenColumns, double[][] weight) {
  super();
  this.visibleColumns = visibleColumns;
  this.hiddenColumns = hiddenColumns;
  this.weight = weight;
  this.visibleBias = Util.zeroArray(visibleColumns.size());
  this.hiddenBias = Util.zeroArray(hiddenColumns.size());
 }
 
 /** このモデルの入力データのカラム定義
  * 
  * @return
  */
 public ColumnInfo getVisibleColumnInfo() {
  return visibleColumns;
 }
 
 /** このモデルの出力データのカラム定義
  * 
  * @return
  */
 public ColumnInfo getHiddenColumnInfo() {
  return hiddenColumns;
 }
 
 /** 重みを加算する。
  * 
  * @param i 重みのインデックス
  * @param j 重みのインデックス
  * @param v 加算値
  */
 public void addWeight(int i, int j, double v) {
  weight[i][j] += v;
 }
 
 /** 可視層バイアスを加算する。
  * 
  * @param index バイアスのインデックス
  * @param v 加算値
  */
 public void addVisibleBias(int index, double v) {
  visibleBias[index] += v;
 }
 
 /** 隠れ層バイアスを加算する。
  * 
  * @param index バイアスのインデックス
  * @param v 加算値
  */
 public void addHiddenBias(int index, double v) {
  hiddenBias[index] += v;
 }
 
 /** propagateUpの結果をシグモイドする。
  * 
  * @param visible 可視層のデータ
  * @return 隠れ層のデータ
  * @throws Exception
  */
 public double[] visible2Hidden(int[] visible) throws Exception {
  return Util.sigmoid(propagateUp(visible));
 }
 
 /** visible2Hiddenのdoubleを引数にとる場合。
  * DBNのpredictで用いている。
  * 
  * @param visible 可視層のデータ
  * @return 隠れ層のデータ
  * @throws Exception
  */
 public double[] visible2Hidden(double[] visible) throws Exception {
  return Util.sigmoid(propagateUp(visible));
 }

 /** propagateDownの結果をシグモイドする。
  * 
  * @param hidden 隠れ層のデータ
  * @return 可視層のデータ
  * @throws Exception
  */
 public double[] hidden2Visible(int[] hidden) throws Exception {
  return Util.sigmoid(propagateDown(hidden));
 }

 /** hidden2Visibleのdoubleを引数にとる場合。
  * 
  * @param hidden 隠れ層のデータ
  * @return 可視層のデータ
  * @throws Exception
  */
 public double[] hidden2Visible(double[] hidden) throws Exception {
  return Util.sigmoid(propagateDown(hidden));
 }

 /** 重みWとvisibleデータの内積を取って、定数項hiddenBiasを加算する。
  * 
  * @param visible 可視層のデータ
  * @return 隠れ層の予測
  * @throws Exception
  */
 public double[] propagateUp(int[] visible) throws Exception {
  double[] hidden = Arrays.copyOf(hiddenBias, hiddenColumns.size());
  for(int i = 0; i < hiddenColumns.size(); ++i) {
   for(int j = 0; j < visibleColumns.size(); ++j) {
    hidden[i] += weight[i][j] * visible[visibleColumns.index(j)];
   }
  }
  return hidden;
 }
 
 /** propagateUpのdoubleを引数にとる場合。
  * DBNのpredictで用いている。
  * 
  * @param visible 可視層のデータ
  * @return 隠れ層の予測
  * @throws Exception
  */
 public double[] propagateUp(double[] visible) throws Exception {
  double[] hidden = Arrays.copyOf(hiddenBias, hiddenColumns.size());
  for(int i = 0; i < hiddenColumns.size(); ++i) {
   for(int j = 0; j < visibleColumns.size(); ++j) {
    hidden[i] += weight[i][j] * visible[visibleColumns.index(j)];
   }
  }
  return hidden;
 }

 /** 重みWとhiddenデータの（縦横が逆の）内積を取って、定数項visibleBiasを加算する。
  * 
  * @param hidden 隠れ層のデータ。
  * @return 可視層の予測
  * @throws Exception
  */
 public double[] propagateDown(int[] hidden) throws Exception {
  if(hidden.length != hiddenColumns.size())throw new Exception("invalid hidden size.");
  double[] visible = Arrays.copyOf(visibleBias, visibleColumns.size());
  for(int i = 0; i < visibleColumns.size(); ++i) {
   for(int j = 0; j < hiddenColumns.size(); ++j) {
    visible[i] += weight[j][i] * hidden[hiddenColumns.index(j)];
   }
  }
  return visible;
 }
 
 /** propagateDownのdoubleを引数にとる場合。
  * RBMのreconstructで用いている。
  * 
  * @param hidden 隠れ層のデータ。
  * @return 可視層の予測
  * @throws Exception
  */
 public double[] propagateDown(double[] hidden) throws Exception {
  if(hidden.length != hiddenColumns.size())throw new Exception("invalid hidden size.");
  double[] visible = Arrays.copyOf(visibleBias, visibleColumns.size());
  for(int i = 0; i < visibleColumns.size(); ++i) {
   for(int j = 0; j < hiddenColumns.size(); ++j) {
    visible[i] += weight[j][i] * hidden[hiddenColumns.index(j)];
   }
  }
  return visible;
 }

}
