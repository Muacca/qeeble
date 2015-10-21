package org.dmwp.qeeble.mnist;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.dmwp.qeeble.DumpUtil;
import org.dmwp.qeeble.common.Model;
import org.dmwp.qeeble.common.VectorDense;
import org.dmwp.qeeble.dbn.DeepBeliefNetwork;
import org.dmwp.qeeble.dbn.DeepBeliefNetworkContext;
import org.dmwp.qeeble.lr.LogisticRegressionContext;
import org.dmwp.qeeble.rbm.RestrictedBoltzmannMachineContext;

public class MNISTLearnDBNTest {

 // params
 private static final double pretrain_lr = 0.1;
 private static final int pretraining_epochs = 1;
 private static final double finetune_lr = 0.1;
 private static final int finetune_epochs = 5;

 private static final int outputLayerSize = 10;
 private static final int[] RBMlayerSizes = {784, 300, 100};


 /**Learning by DBN from MNIST data
  * @param args
  *          args[0]: label file
  *          args[1]: image file
  * @throws IOException
  */
 public static void main(String[] args) {
  if(args.length != 2) {
   System.err.println("usage: <label file> <image file>");
   System.exit(1);
  }

  long start = System.currentTimeMillis();
  try{
   DeepBeliefNetworkContext context = new DeepBeliefNetworkContext(
    new RestrictedBoltzmannMachineContext(new Random(1234), pretrain_lr, pretraining_epochs),
    new LogisticRegressionContext(finetune_lr, finetune_epochs));

   // pretrain
   System.out.println("pretrain start.");
   List<Model> preModels = new ArrayList<Model>();
   for(int i = 0; i < RBMlayerSizes.length - 1; ++i) {
    System.out.println("layer: " + (i + 1));
    Model model = context.getPreContext().create(RBMlayerSizes[i], RBMlayerSizes[i + 1]);
    while(context.getPreContext().hasNext()) {
     System.out.print("epoch: " + context.getPreContext().currentEpoch());
     pretrain(context, model, preModels, args[0], args[1]);
     context.getPreContext().next();
    }
    System.out.println((System.currentTimeMillis() - start) + "msec");
    preModels.add(model);
    context.getPreContext().initEpoch();  
   }

   // finetune
   System.out.println("finetune start.");
   Model outputModel = Model.createEmpty(RBMlayerSizes[RBMlayerSizes.length - 1], outputLayerSize);
   while(context.getFineContext().hasNext()) {
    System.out.print("epoch: " + context.getFineContext().currentEpoch());
    finetune(context, outputModel, preModels, args[0], args[1]);
    context.getFineContext().next();
   }

   // predict
   System.out.println("predict start.");
   predict(outputModel, preModels, args[0], args[1]);

  }catch(Exception e) {
   e.printStackTrace();
   System.exit(1);
  }
 }

 private static void pretrain(DeepBeliefNetworkContext context, Model model, List<Model> preModels, String labelFilename, String imageFilename) throws Exception {
  try(
   DataInputStream labelInput = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilename)));
   DataInputStream imageInput = new DataInputStream(new BufferedInputStream(new FileInputStream(imageFilename)));
   ){
   MNISTReader in = MNISTReader.create(labelInput, imageInput);
   double[] buf = in.getInfo().createDoubleBuffer();
   while(in.available()) {
    MNISTDataDouble data = in.readDouble(buf);
    DeepBeliefNetwork.pretrain(context, model, preModels, VectorDense.create(data.getBuf()));
    if(in.readCount() % 200 == 0) {
     System.out.print(".");
    }
   }
  }
 }

 private static void finetune(DeepBeliefNetworkContext context, Model outputModel, List<Model> preModels, String labelFilename, String imageFilename) throws Exception {
  try(
   DataInputStream labelInput = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilename)));
   DataInputStream imageInput = new DataInputStream(new BufferedInputStream(new FileInputStream(imageFilename)));
   ){
   MNISTReader in = MNISTReader.create(labelInput, imageInput);
   double[] buf = in.getInfo().createDoubleBuffer();
   double[] result = new double[10];
   while(in.available()) {
    MNISTDataDouble data = in.readDouble(buf);
    setupResult(result, data.getLabel());
    DeepBeliefNetwork.finetune(context, outputModel, preModels, VectorDense.create(data.getBuf()), VectorDense.create(result));
    if(in.readCount() % 200 == 0) {
     System.out.print(".");
    }
   }
  }
 }
 
 private static void predict(Model outputModel, List<Model> preModels, String labelFilename, String imageFilename) throws Exception {
  try(
   DataInputStream labelInput = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilename)));
   DataInputStream imageInput = new DataInputStream(new BufferedInputStream(new FileInputStream(imageFilename)));
   ){
   MNISTReader in = MNISTReader.create(labelInput, imageInput);
   double[] buf = in.getInfo().createDoubleBuffer();
   while(in.available()) {
    MNISTDataDouble data = in.readDouble(buf);
    System.out.print(data.getLabel() + " --> ");
    DumpUtil.dump(DeepBeliefNetwork.predict(outputModel, preModels, VectorDense.create(data.getBuf())));
   }
  }
 }

 private static void setupResult(double[] result, int label) {
  for(int i = 0; i < result.length; ++i) {
   result[i] = (i == label ? 1 : 0);
  }
 }

}
