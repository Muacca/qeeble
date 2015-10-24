package org.dmwp.qeeble.mnist;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.dmwp.qeeble.DumpUtil;
import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.Vector;
import org.dmwp.qeeble.common.VectorDense;
import org.dmwp.qeeble.dbn.DeepBeliefNetwork;
import org.dmwp.qeeble.dbn.DeepBeliefNetworkContext;
import org.dmwp.qeeble.lr.LogisticRegressionContext;
import org.dmwp.qeeble.rbm.RestrictedBoltzmannMachineContext;

public class MNISTLearnDBNTest {

 // params
 private static final double pretrain_lr = 0.1;
 private static final int pretraining_epochs = 10;
 private static final double finetune_lr = 0.1;
 private static final int finetune_epochs = 100;

 private static final int outputLayerSize = 10;
 private static final int[] RBMlayerSizes = {784, 196, 49};


 /**Learning by DBN from MNIST data
  * @param args
  *          args[0]: label file for train
  *          args[1]: image file for train
  *          args[2]: label file for predict
  *          args[3]: image file for predict
  * @throws IOException
  */
 public static void main(String[] args) {
  if(args.length != 4) {
   System.err.println("usage: <label file for train> <image file for train> <label file for predict> <image file for predict>");
   System.exit(1);
  }
  
  DeepBeliefNetworkContext context = new DeepBeliefNetworkContext(
   new RestrictedBoltzmannMachineContext(new Random(1234), pretrain_lr, pretraining_epochs),
   new LogisticRegressionContext(finetune_lr, finetune_epochs));


  long start = System.currentTimeMillis();

  // load MNIST data
  int[] labels = null;
  Vector[] results = null;
  Vector[] images = null; 
  System.out.println("load start.");
  try(
   DataInputStream labelInput = new DataInputStream(new BufferedInputStream(new FileInputStream(args[0])));
   DataInputStream imageInput = new DataInputStream(new BufferedInputStream(new FileInputStream(args[1])));
   ){
   MNISTReader in = MNISTReader.create(labelInput, imageInput);
   labels = new int[in.getInfo().getSize()];
   results = new Vector[in.getInfo().getSize()];
   images = new Vector[in.getInfo().getSize()]; 
   int line = 0;
   while(in.available()) {
    MNISTDataDouble data = in.readDouble();
    labels[line] = data.getLabel();
    results[line] = setupResult(labels[line]);
    images[line] = context.getPreContext().binomial(VectorDense.create(data.getBuf()));
    line++;
    if(line % 1000 == 0) {
     System.out.print(".");
    }
   }
  } catch(Exception e) {
   e.printStackTrace();
   System.exit(1);
  }
  System.out.println((System.currentTimeMillis() - start) + "msec");

  if(labels == null) {
   System.out.println("load failed.");
   System.exit(1);
  }
  
  List<Model> preModels = new ArrayList<Model>();
  Model outputModel = Model.createEmpty(RBMlayerSizes[RBMlayerSizes.length - 1], outputLayerSize);
  try {
   // pretrain
   System.out.println("pretrain start.");
   for(int i = 0; i < RBMlayerSizes.length - 1; ++i) {
    System.out.println("layer: " + (i + 1));
    Model model = context.getPreContext().create(RBMlayerSizes[i], RBMlayerSizes[i + 1]);
    while(context.getPreContext().hasNext()) {
     System.out.print("epoch: " + (context.getPreContext().currentEpoch() + 1));
     pretrain(context, model, preModels, images);
     context.getPreContext().next();
     System.out.println((System.currentTimeMillis() - start) + "msec");
    }
    preModels.add(model);
    context.getPreContext().initEpoch();  
   }

   // finetune
   System.out.println("finetune start.");
   while(context.getFineContext().hasNext()) {
    System.out.print("epoch: " + (context.getFineContext().currentEpoch() + 1));
    finetune(context, outputModel, preModels, results, images);
    context.getFineContext().next();
    System.out.println((System.currentTimeMillis() - start) + "msec");
   }
  }catch(Exception e) {
   e.printStackTrace();
   System.exit(1);
  }

   try(
    PrintStream out = new PrintStream(new FileOutputStream("./log.txt"));
    DataInputStream labelInput = new DataInputStream(new BufferedInputStream(new FileInputStream(args[2])));
    DataInputStream imageInput = new DataInputStream(new BufferedInputStream(new FileInputStream(args[3])));
    ){
   // predict
   System.out.println("predict start.");
   MNISTReader in = MNISTReader.create(labelInput, imageInput);
   int correctCount = predict(out, outputModel, preModels, in);
   System.out.println("correct:" + correctCount + "/" + in.getInfo().getSize() + " (" + ((double)correctCount * 100) / in.getInfo().getSize() + "%)");
   System.out.println((System.currentTimeMillis() - start) + "msec");
  }catch(Exception e) {
   e.printStackTrace();
   System.exit(1);
  }
 }

 private static void pretrain(DeepBeliefNetworkContext context, Model model, List<Model> preModels, Vector[] images) throws Exception {
  for(int i = 0; i < images.length; ++i) {
   DeepBeliefNetwork.pretrain(context, model, preModels, images[i]);
   if(i % 1000 == 0) {
    System.out.print(".");
   }
  }
 }

 private static void finetune(DeepBeliefNetworkContext context, Model outputModel, List<Model> preModels, Vector[] results, Vector[] images) throws Exception {
  for(int i = 0; i < results.length; ++i) {
   DeepBeliefNetwork.finetune(context, outputModel, preModels, images[i], results[i]);
   if(i % 1000 == 0) {
    System.out.print(".");
   }
  }
 }

 private static int predict(PrintStream out, Model outputModel, List<Model> preModels, MNISTReader in) throws Exception {
  int correct = 0;
  while(in.available()) {
   MNISTDataDouble data = in.readDouble();
   Vector result = DeepBeliefNetwork.predict(outputModel, preModels, VectorDense.create(data.getBuf()));
   int answer = predictResult(result);
   out.print((answer == data.getLabel() ? 1 : 0) + ":" + data.getLabel() + ":" + answer + " --> ");
   DumpUtil.dump(out, result);
   correct += (answer == data.getLabel() ? 1 : 0);
  }
  return correct;
 }

 private static Vector setupResult(int label) {
  double[] result = new double[10];
  for(int i = 0; i < result.length; ++i) {
   result[i] = (i == label ? 1 : 0);
  }
  return VectorDense.create(result);
 }

 private static int predictResult(Vector result) {
  int label = 0;
  double max = -1.0;
  for(int i = 0; i < result.size(); ++i) {
   if(result.get(i) > max) {
    max = result.get(i);
    label = i;
   }
  }
  return label;
 }
 
}
