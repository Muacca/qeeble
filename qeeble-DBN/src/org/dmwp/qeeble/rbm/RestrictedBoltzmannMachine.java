package org.dmwp.qeeble.rbm;

import java.util.Random;

import org.dmwp.qeeble.Model;
import org.dmwp.qeeble.Util;

public class RestrictedBoltzmannMachine {

 public static void train(Model model, Random rng, int[] input, double lr, int cdSteps) throws Exception {
  /* CD-1 */
  double[] ph_mean = model.visible2Hidden(input);
  int[] ph_sample = Util.binomial(rng, ph_mean);

  /* CD-k */
  double[] nh_means = ph_mean;
  int[] nv_samples = null;
  for(int step = 0; step < cdSteps; step++) {
   nv_samples = Util.binomial(rng, model.hidden2Visible(Util.binomial(rng, nh_means)));
   nh_means = model.visible2Hidden(nv_samples);
  }
  if(nv_samples == null)throw new Exception("invalid cdSteps.");
  
  for(int i = 0; i < model.getHiddenColumnInfo().size(); i++) {
   for(int j = 0; j < model.getVisibleColumnInfo().size(); j++) {
    model.addWeight(i, j, lr * (ph_mean[i] * input[j] - nh_means[i] * nv_samples[j]));
   }
   model.addHiddenBias(i, lr * (ph_sample[i] - nh_means[i]));
  }

  for(int i = 0; i < model.getVisibleColumnInfo().size(); i++) {
   model.addVisibleBias(i, lr * (input[i] - nv_samples[i]));
  }

 }

 public static double[] reconstruct(Model model, int[] input) throws Exception {
  return model.hidden2Visible(model.visible2Hidden(input));
 }

 private RestrictedBoltzmannMachine() {}

}
