package org.dmwp.qeeble.common;

import java.util.Random;

public class Model {
 
 /** uniformedされた重み行列を持つモデルを作成する。
  * 
  * @param rng
  * @param columnInfo
  * @return
  */
 public static Model createUniformed(Random rng, int visibleColumnSize, int hiddenColumnSize) {
  return new Model(MatrixDense.createUniformed(rng, hiddenColumnSize, visibleColumnSize));
 }
 
 /** ゼロ行列を重みとして持つモデルを作成する。
  * 
  * @param columnInfo
  * @return
  */
 public static Model createEmpty(int visibleColumnSize, int hiddenColumnSize) {
  return new Model(MatrixDense.createEmpty(hiddenColumnSize, visibleColumnSize));
 }

 private Matrix weight;
 private Vector visibleBias;
 private Vector hiddenBias;
 private Model(Matrix weight) {
  super();
  this.weight = weight;
  this.visibleBias = VectorDense.createEmpty(weight.columnSize());
  this.hiddenBias = VectorDense.createEmpty(weight.rowSize());
 }
 
 /** このモデルの入力データのカラム数
  * 
  * @return
  */
 public int getVisibleColumnSize() {
  return weight.columnSize();
 }
 
 /** このモデルの出力データのカラム数
  * 
  * @return
  */
 public int getHiddenColumnSize() {
  return weight.rowSize();
 }
 
 /** 重みを加算する。
  * 
  * @param i 重みのインデックス
  * @param j 重みのインデックス
  * @param v 加算値
  */
 public void addWeight(int i, int j, double v) {
  weight.add(i, j, v);
 }
 
 /** 可視層バイアスを加算する。
  * 
  * @param index バイアスのインデックス
  * @param v 加算値
  */
 public void addVisibleBias(int index, double v) {
  visibleBias.add(index, v);
 }
 
 /** 隠れ層バイアスを加算する。
  * 
  * @param index バイアスのインデックス
  * @param v 加算値
  */
 public void addHiddenBias(int index, double v) {
  hiddenBias.add(index, v);
 }
 
 /** propagateUpの結果をシグモイドする。
  * 
  * @param visible 可視層のデータ
  * @return 隠れ層のデータ
  * @throws Exception
  */
 public Vector visible2Hidden(Vector visible) throws Exception {
  return propagateUp(visible).sigmoid();
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
 public Vector hidden2Visible(Vector hidden) throws Exception {
  return propagateDown(hidden).sigmoid();
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
  if(visible.length != getVisibleColumnSize())throw new Exception("invalid visible size.");
  double[] hidden = hiddenBias.copyArray();
  for(int i = 0; i < getHiddenColumnSize(); ++i) {
   for(int j = 0; j < getVisibleColumnSize(); ++j) {
    hidden[i] += weight.get(i, j) * visible[j];
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
  if(visible.length != getVisibleColumnSize())throw new Exception("invalid visible size.");
  double[] hidden = hiddenBias.copyArray();
  for(int i = 0; i < getHiddenColumnSize(); ++i) {
   for(int j = 0; j < getVisibleColumnSize(); ++j) {
    hidden[i] += weight.get(i, j) * visible[j];
   }
  }
  return hidden;
 }

 /** propagateUpのVectorを引数にとる場合。
  * DBNのpredictで用いている。
  * 
  * @param visible 可視層のデータ
  * @return 隠れ層の予測
  * @throws Exception
  */
 public Vector propagateUp(Vector visible) throws Exception {
  if(visible.size() != getVisibleColumnSize())throw new Exception("invalid visible size.");
  Vector hidden = hiddenBias.copy();
  for(int i = 0; i < getHiddenColumnSize(); ++i) {
   for(int j = 0; j < getVisibleColumnSize(); ++j) {
    hidden.add(i, weight.get(i, j) * visible.get(j));
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
  if(hidden.length != getHiddenColumnSize())throw new Exception("invalid hidden size.");
  double[] visible = visibleBias.copyArray();
  for(int i = 0; i < getVisibleColumnSize(); ++i) {
   for(int j = 0; j < getHiddenColumnSize(); ++j) {
    visible[i] += weight.get(j, i) * hidden[j];
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
  if(hidden.length != getHiddenColumnSize())throw new Exception("invalid hidden size.");
  double[] visible = visibleBias.copyArray();
  for(int i = 0; i < getVisibleColumnSize(); ++i) {
   for(int j = 0; j < getHiddenColumnSize(); ++j) {
    visible[i] += weight.get(j, i) * hidden[j];
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
 public Vector propagateDown(Vector hidden) throws Exception {
  if(hidden.size() != getHiddenColumnSize())throw new Exception("invalid hidden size.");
  Vector visible = visibleBias.copy();
  for(int i = 0; i < getVisibleColumnSize(); ++i) {
   for(int j = 0; j < getHiddenColumnSize(); ++j) {
    visible.add(i, weight.get(j, i) * hidden.get(j));
   }
  }
  return visible;
 }

}
