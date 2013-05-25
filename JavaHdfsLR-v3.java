import spark.api.java.JavaRDD;
import spark.api.java.JavaSparkContext;
import spark.api.java.function.Function;
import spark.api.java.function.Function2;

import java.io.*;
import java.util.*;

public class JavaHdfsLR {
	static int D = 5000;
	static Random rand = new Random(42);
	
	static class Xvector implements Serializable {
		Vector<Integer> x_index;
		Vector<Double> x_value;
		public Xvector(Vector<Integer> x_index, Vector<Double> x_value) {
			this.x_index = x_index;
	    	this.x_value = x_value;
		}
	}
	
	static class DataPoint implements Serializable {
		Xvector x;
	    double y;
	    public DataPoint(Xvector x, double y) {
	    	this.x = x;
	    	this.y = y;
	    }
	}

	static class ParsePoint extends Function<String, DataPoint> {
		public DataPoint call(String line) {
			Vector<Integer> x_index=new Vector<Integer>();
			Vector<Double> x_value=new Vector<Double>();
			x_index.clear();
			x_value.clear();
			StringTokenizer itr = new StringTokenizer(line, " ");
			double y = Double.parseDouble(itr.nextToken());
			String tmp=itr.nextToken();
			while (tmp.contains(":")) {
				String[] strs=tmp.split(":");
				x_index.addElement(Integer.parseInt(strs[0]));
				x_value.addElement(Double.parseDouble(strs[1]));
				if (itr.hasMoreTokens()) tmp=itr.nextToken();
				else break;
			}
			return new DataPoint(new Xvector(x_index,x_value),y);
		}
	}

	static class VectorSum extends Function2<DataPoint, DataPoint, DataPoint> {
		public DataPoint call(DataPoint a, DataPoint b) {
			DataPoint result = new DataPoint(new Xvector(new Vector<Integer>(),new Vector<Double>()), 0);
			result.x.x_index.clear();
			result.x.x_value.clear();
			
			/*
			int num1=a.x.x_index.size();
			int num2=b.x.x_index.size();
			for (int i=0;i<num1;i++) {
				result.x.x_index.addElement(a.x.x_index.elementAt(i));
				result.x.x_value.addElement(a.x.x_value.elementAt(i));
			}
			for (int i=0;i<num2;i++) {
				int p=result.x.x_index.indexOf(b.x.x_index.elementAt(i));
				if (p==-1) {
					result.x.x_index.addElement(b.x.x_index.elementAt(i));
					result.x.x_value.addElement(b.x.x_value.elementAt(i));
				}
				else result.x.x_value.set(p,result.x.x_value.elementAt(p) + b.x.x_value.elementAt(i));
			}
			*/
			
			int num1=a.x.x_index.size();
			int num2=b.x.x_index.size();
			int p1=0;
			int p2=0;
			while (p1<num1||p2<num2) {
				int index1,index2;
				if (p1==num1) index1=Integer.MAX_VALUE;
				else index1=a.x.x_index.elementAt(p1);
				if (p2==num2) index2=Integer.MAX_VALUE;
				else index2=b.x.x_index.elementAt(p2);
				if (p1>=num1&&p2>=num2) break;
				if (index1<index2) {
					result.x.x_index.addElement(index1);
					result.x.x_value.addElement(a.x.x_value.elementAt(p1));
					p1++;
				}
				if (index1==index2) {
					result.x.x_index.addElement(index1);
					result.x.x_value.addElement(a.x.x_value.elementAt(p1)+b.x.x_value.elementAt(p2));
					p1++;
					p2++;
				}
				if (index1>index2) {
					result.x.x_index.addElement(index2);
					result.x.x_value.addElement(b.x.x_value.elementAt(p2));
					p2++;
				}
			}
			
			result.y=a.y+b.y;
			return result;
		}
	}

	static class ComputeGradient extends Function<DataPoint, DataPoint> {
		double[] weights;
		double b;
		int t;
		
		public ComputeGradient(double[] weights,double b,int t) {
			this.weights = weights;
			this.b = b;
			this.t = t;
		}
		
		public DataPoint call(DataPoint p) throws Exception {
			DataPoint gradient = new DataPoint(new Xvector(new Vector<Integer>(),new Vector<Double>()), 0);
			gradient.x.x_index.clear();
			gradient.x.x_value.clear();
			Xvector curx=p.x;
			int num=curx.x_index.size();
			double tmp_value = (1 / (1 + Math.exp(-p.y * (dot(weights, curx)+b))) - 1) * p.y;
			//for (int i = 0; i < num; i++)
			//	gradient.x.x_index.addElement(curx.x_index.elementAt(i));
			gradient.x.x_index=curx.x_index;
			for (int i = 0; i < num; i++)
				gradient.x.x_value.addElement(tmp_value*curx.x_value.elementAt(i));
			gradient.y=tmp_value;
			return gradient;
		}
	}

	public static double dot(double[] a, Xvector x) {
		double res = 0;
		int num=x.x_index.size();
		for (int i = 0; i < num; i++) {
			int index=x.x_index.elementAt(i);
			res += a[index-1] * x.x_value.elementAt(i);
		}
		return res;
	}

	public static void printWeights(double[] a, double b) throws Exception {
		File fout=new File("output-model.txt");
		FileWriter writer = new FileWriter(fout);
		BufferedWriter bw= new BufferedWriter(writer);
		for (int i=0;i<D;i++) {
			bw.write(a[i]+"\n");
			bw.flush();
		}
		bw.write(b+"\n");
		bw.flush();
		bw.close();
		writer.close();
	}

	public static void main(String[] args) throws Exception {
		if (args.length < 3) {
			System.err.println("Usage: JavaHdfsLR <master> <file> <iters>");
			System.exit(1);
		}

		JavaSparkContext sc = new JavaSparkContext(args[0], "JavaHdfsLR", System.getenv("SPARK_HOME"), System.getenv("SPARK_EXAMPLES_JAR"));
		JavaRDD<String> lines = sc.textFile(args[1]);
		JavaRDD<DataPoint> points = lines.map(new ParsePoint()).cache();
		int ITERATIONS = Integer.parseInt(args[2]);

		double[] w = new double[D];
		for (int i = 0; i < D; i++)
			w[i] = 0;
		double b=0;

		for (int i = 1; i <= ITERATIONS; i++) {
			System.out.println(w[0]);
			System.out.println(w[1]);
			System.out.println(b);
			System.out.println("On iteration " + i);
			DataPoint gradient = points.map(new ComputeGradient(w,b,i)).reduce(new VectorSum());
			int num=gradient.x.x_index.size();
			for (int j = 0; j < num; j++) {
				int index=gradient.x.x_index.elementAt(j);
				w[index-1] -= gradient.x.x_value.elementAt(j);
			}
			b -= gradient.y;
			/*
			double res = 0;
			for (int j = 0; j < D; j++)
				res = res + w[j] * w[j];
			//res = res + b * b;
			res = Math.sqrt(res);
			
			*/
			/*
			for (int j = 0; j < D; j++)
				w[j] = w[j] / 200;
			b = b / 200;
			*/
		}

		System.out.println("All Iterations Completed!");
		printWeights(w,b);
		System.exit(0);
	}
}