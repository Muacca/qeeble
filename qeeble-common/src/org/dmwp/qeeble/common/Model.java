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
 
 /** 重みを加算する。
  * 
  * @param i 重みのインデックス
  * @param j 重みのインデックス
  * @param v 加算値
  */
 public void addWeight(int i, int j, double v) {
  weight.add(i, j, v);
 }
 
 /** 重みを加算する。
  * 
  * @param m 加算値
  */
 public void addWeight(Matrix m) throws Exception {
  weight.add(m);
 }
 
 /** 可視層バイアスを加算する。
  * 
  * @param index バイアスのインデックス
  * @param v 加算値
  */
 public void addVisibleBias(int index, double v) {
  visibleBias.add(index, v);
 }
 
 /** 可視層バイアスを加算する。
  * 
  * @param v 加算値
  */
 public void addVisibleBias(Vector v) throws Exception {
  visibleBias.add(v);
 }

 /** 隠れ層バイアスを加算する。
  * 
  * @param index バイアスのインデックス
  * @param v 加算値
  */
 public void addHiddenBias(int index, double v) {
  hiddenBias.add(index, v);
 }
 
 /** 隠れ層バイアスを加算する。
  * 
  * @param v 加算値
  */
 public void addHiddenBias(Vector v) throws Exception {
  hiddenBias.add(v);
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
 
 /** propagateDownの結果をシグモイドする。
  * 
  * @param hidden 隠れ層のデータ
  * @return 可視層のデータ
  * @throws Exception
  */
 public Vector hidden2Visible(Vector hidden) throws Exception {
  return propagateDown(hidden).sigmoid();
 }

 /** propagateUpのVectorを引数にとる場合。
  * DBNのpredictで用いている。
  * 
  * @param visible 可視層のデータ
  * @return 隠れ層の予測
  * @throws Exception
  */
 public Vector propagateUp(Vector visible) throws Exception {
  Vector hidden = weight.innerProduct(visible);
  hidden.add(hiddenBias);
  return hidden;
 }

 /** propagateDownのdoubleを引数にとる場合。
  * RBMのreconstructで用いている。
  * 
  * @param hidden 隠れ層のデータ。
  * @return 可視層の予測
  * @throws Exception
  */
 public Vector propagateDown(Vector hidden) throws Exception {
  Vector visible = weight.innerProductFrom(hidden);
  visible.add(visibleBias);
  return visible;
 }

}
