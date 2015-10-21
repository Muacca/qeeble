package org.dmwp.qeeble;

import java.io.PrintStream;

import org.dmwp.qeeble.common.Vector;

public class DumpUtil {
 
 public static final String DEFAULT_SEPARATOR = "\t";

 public static void dump(Vector v) {
  dump(System.out, v);
 }
 public static void dump(Vector array, String separator) {
  dump(System.out, array, separator);
 }
 
 public static void dump(PrintStream out, Vector array) {
  dump(out, array, DEFAULT_SEPARATOR);
 }
 public static void dump(PrintStream out, Vector v, String separator) {
  for(double t: v) {
   out.print(t);
   out.print(separator);
  }
  out.println();
 }


 public static void dump(double[] array) {
  dump(System.out, array);
 }
 
 public static void dump(double[] array, String separator) {
  dump(System.out, array, separator);
 }
 
 public static void dump(PrintStream out, double[] array) {
  dump(out, array, DEFAULT_SEPARATOR);
 }
 public static void dump(PrintStream out, double[] array, String separator) {
  for(double t: array) {
   out.print(t);
   out.print(separator);
  }
  out.println();
 }
 
 public static void dump(byte[] array) {
  dump(System.out, array);
 }
 
 public static void dump(byte[] array, String separator) {
  dump(System.out, array, separator);
 }
 
 public static void dump(PrintStream out, byte[] array) {
  dump(out, array, DEFAULT_SEPARATOR);
 }
 public static void dump(PrintStream out, byte[] array, String separator) {
  for(double t: array) {
   out.print(t);
   out.print(separator);
  }
  out.println();
 }

 public static void separatorLine() {
  separatorLine(System.out);
 }

 public static void separatorLine(PrintStream out) {
  out.println("============");
 }
 
 private DumpUtil(){}
}
