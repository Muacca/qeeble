package org.dmwp.qeeble;

import java.io.PrintStream;

public class DumpUtil {
 
 public static final String DEFAULT_SEPARATOR = "\t";

 public static void dumpArray(double[] array) {
  dumpArray(System.out, array);
 }
 
 public static void dumpArray(double[] array, String separator) {
  dumpArray(System.out, array, separator);
 }
 
 public static void dumpArray(PrintStream out, double[] array) {
  dumpArray(out, array, DEFAULT_SEPARATOR);
 }
 public static void dumpArray(PrintStream out, double[] array, String separator) {
  for(double t: array) {
   out.print(t);
   out.print(separator);
  }
  out.println();
 }
 
 private DumpUtil(){}
}
