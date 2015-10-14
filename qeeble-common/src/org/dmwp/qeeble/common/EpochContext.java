package org.dmwp.qeeble.common;

public abstract class EpochContext {

 private int epochMax;
 private int epochCount;
 protected EpochContext(int epochMax) {
  super();
  this.epochMax = epochMax;
  this.epochCount = 0;
 }
 
 public boolean hasNext() {
  return (epochCount < epochMax);
 }
 
 public void next() {
  epochCount++;
 }
 
 public int currentEpoch() {
  return epochCount;
 }
 
 public int maxEpoch() {
  return epochMax;
 }
 
 public void initEpoch() {
  epochCount = 0;
 }
}
