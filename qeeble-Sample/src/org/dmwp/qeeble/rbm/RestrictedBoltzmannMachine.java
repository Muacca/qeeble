package org.dmwp.qeeble.rbm;

import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.Vector;

public class RestrictedBoltzmannMachine {

 public static void train(RestrictedBoltzmannMachineContext context, Model model, Vector input) throws Exception {
  Vector ph_mean = model.visible2Hidden(input);
  Vector ph_sample = context.binomial(ph_mean);

  /* CD-k */
  Vector nh_means = ph_mean;
  Vector nv_samples = null;
  for(int step = 0; step < context.getContrastiveDivergenceStep(); ++step) {
   nv_samples = context.binomial(model.hidden2Visible(context.binomial(nh_means)));
   nh_means = model.visible2Hidden(nv_samples);
  }
  if(nv_samples == null)throw new Exception("invalid cdSteps.");
  
  for(int i = 0; i < model.getHiddenColumnSize(); ++i) {
   for(int j = 0; j < model.getVisibleColumnSize(); ++j) {
    model.addWeight(i, j, context.getLearningRate() * (ph_mean.get(i) * input.get(j) - nh_means.get(i) * nv_samples.get(j)));
   }
   model.addHiddenBias(i, context.getLearningRate() * (ph_sample.get(i) - nh_means.get(i)));
  }

  for(int j = 0; j < model.getVisibleColumnSize(); ++j) {
   model.addVisibleBias(j, context.getLearningRate() * (input.get(j) - nv_samples.get(j)));
  }

 }

 public static void train(RestrictedBoltzmannMachineContext context, Model model, double[] input) throws Exception {
  double[] ph_mean = model.visible2Hidden(input);
  int[] ph_sample = context.binomial(ph_mean);

  /* CD-k */
  double[] nh_means = ph_mean;
  int[] nv_samples = null;
  for(int step = 0; step < context.getContrastiveDivergenceStep(); ++step) {
   nv_samples = context.binomial(model.hidden2Visible(context.binomial(nh_means)));
   nh_means = model.visible2Hidden(nv_samples);
  }
  if(nv_samples == null)throw new Exception("invalid cdSteps.");
  
  for(int i = 0; i < model.getHiddenColumnSize(); ++i) {
   for(int j = 0; j < model.getVisibleColumnSize(); ++j) {
    model.addWeight(i, j, context.getLearningRate() * (ph_mean[i] * input[j] - nh_means[i] * nv_samples[j]));
   }
   model.addHiddenBias(i, context.getLearningRate() * (ph_sample[i] - nh_means[i]));
  }

  for(int j = 0; j < model.getVisibleColumnSize(); ++j) {
   model.addVisibleBias(j, context.getLearningRate() * (input[j] - nv_samples[j]));
  }

 }

 public static void train(RestrictedBoltzmannMachineContext context, Model model, int[] input) throws Exception {
  double[] ph_mean = model.visible2Hidden(input);
  int[] ph_sample = context.binomial(ph_mean);

  /* CD-k */
  double[] nh_means = ph_mean;
  int[] nv_samples = null;
  for(int step = 0; step < context.getContrastiveDivergenceStep(); ++step) {
   nv_samples = context.binomial(model.hidden2Visible(context.binomial(nh_means)));
   nh_means = model.visible2Hidden(nv_samples);
  }
  if(nv_samples == null)throw new Exception("invalid cdSteps.");
  
  for(int i = 0; i < model.getHiddenColumnSize(); ++i) {
   for(int j = 0; j < model.getVisibleColumnSize(); ++j) {
    model.addWeight(i, j, context.getLearningRate() * (ph_mean[i] * input[j] - nh_means[i] * nv_samples[j]));
   }
   model.addHiddenBias(i, context.getLearningRate() * (ph_sample[i] - nh_means[i]));
  }

  for(int j = 0; j < model.getVisibleColumnSize(); ++j) {
   model.addVisibleBias(j, context.getLearningRate() * (input[j] - nv_samples[j]));
  }

 }

 public static double[] reconstruct(Model model, int[] input) throws Exception {
  return model.hidden2Visible(model.visible2Hidden(input));
 }

 public static double[] reconstruct(Model model, double[] input) throws Exception {
  return model.hidden2Visible(model.visible2Hidden(input));
 }

 public static Vector reconstruct(Model model, Vector input) throws Exception {
  return model.hidden2Visible(model.visible2Hidden(input));
 }

 private RestrictedBoltzmannMachine() {}

}
