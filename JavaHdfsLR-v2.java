import spark.api.java.JavaRDD;
import spark.api.java.JavaSparkContext;
import spark.api.java.function.Function;
import spark.api.java.function.Function2;

import java.io.*;
import java.util.*;

public class JavaHdfsLR {
	static int D = 16428;
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
		Vector<Integer> x_index=new Vector<Integer>();
		Vector<Double> x_value=new Vector<Double>();
		public DataPoint call(String line) {
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

	static class VectorSum extends Function2<Xvector, Xvector, Xvector> {
		public Xvector call(Xvector a, Xvector b) {
			Xvector result = new Xvector(new Vector<Integer>(),new Vector<Double>());
			result.x_index.clear();
			result.x_value.clear();
			int num1=a.x_index.size();
			int num2=b.x_index.size();
			int p1=0;
			int p2=0;
			while (p1<num1||p2<num2) {
				int index1,index2;
				if (p1==num1) index1=Integer.MAX_VALUE;
				else index1=a.x_index.elementAt(p1);
				if (p2==num2) index2=Integer.MAX_VALUE;
				else index2=b.x_index.elementAt(p2);
				if (p1>=num1&&p2>=num2) break;
				if (index1<index2) {
					result.x_index.addElement(index1);
					result.x_value.addElement(a.x_value.elementAt(p1));
					p1++;
				}
				if (index1==index2) {
					result.x_index.addElement(index1);
					result.x_value.addElement(a.x_value.elementAt(p1)+b.x_value.elementAt(p2));
					p1++;
					p2++;
				}
				if (index1>index2) {
					result.x_index.addElement(index2);
					result.x_value.addElement(b.x_value.elementAt(p2));
					p2++;
				}
			}
			return result;
		}
	}

	static class ComputeGradient extends Function<DataPoint, Xvector> {
		double[] weights;
		
		public ComputeGradient(double[] weights) {
			this.weights = weights;
		}
		
		public Xvector call(DataPoint p) {
			Xvector gradient = new Xvector(new Vector<Integer>(),new Vector<Double>());
			Xvector curx=p.x;
			int num=curx.x_index.size();
			double tmp_value = (1 / (1 + Math.exp(-p.y * dot(weights, curx))) - 1) * p.y;
			gradient.x_index=curx.x_index;
			for (int i = 0; i < num; i++)
				gradient.x_value.addElement(tmp_value*curx.x_value.elementAt(i)) ;
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

	public static void printWeights(double[] a) {
		//System.out.println(Arrays.toString(a));
	}

	public static void main(String[] args) {
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
			w[i] = 2 * rand.nextDouble() - 1;

		System.out.print("Initial w: ");
		printWeights(w);

		for (int i = 1; i <= ITERATIONS; i++) {
			System.out.println("On iteration " + i);
			Xvector gradient = points.map(new ComputeGradient(w)).reduce(new VectorSum());
			int num=gradient.x_index.size();
			for (int j = 0; j < num; j++) {
				int index=gradient.x_index.elementAt(j);
				w[index-1] -= gradient.x_value.elementAt(j);
			}
		}

		System.out.print("Final w: ");
		printWeights(w);
		System.exit(0);
	}
}