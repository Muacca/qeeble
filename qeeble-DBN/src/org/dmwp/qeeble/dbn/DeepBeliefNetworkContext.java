package org.dmwp.qeeble.dbn;

import org.dmwp.qeeble.lr.LogisticRegressionContext;
import org.dmwp.qeeble.rbm.RestrictedBoltzmannMachineContext;

public class DeepBeliefNetworkContext {

 private RestrictedBoltzmannMachineContext preContext;
 private LogisticRegressionContext fineContext;
 public DeepBeliefNetworkContext(RestrictedBoltzmannMachineContext preContext, LogisticRegressionContext fineContext) {
  super();
  this.preContext = preContext;
  this.fineContext = fineContext;
 }
 public RestrictedBoltzmannMachineContext getPreContext() {
  return preContext;
 }
 public LogisticRegressionContext getFineContext() {
  return fineContext;
 }
}
