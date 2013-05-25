import spark.api.java.JavaRDD;
import spark.api.java.JavaSparkContext;
import spark.api.java.function.Function;
import spark.api.java.function.Function2;

import java.io.*;
import java.util.*;

class Xvector implements Serializable {
	Vector<Integer> x_index;
	Vector<Double> x_value;
	public Xvector(Vector<Integer> x_index, Vector<Double> x_value) {
		this.x_index = x_index;
		this.x_value = x_value;
	}
}

class DataPoint implements Serializable {
	int line_num;
	Xvector x;
	double y;
	public DataPoint(int line_num, Xvector x, double y) {
		this.line_num = line_num;
		this.x = x;
		this.y = y;
	}
}

class ParsePoint extends Function<String, DataPoint> {
	public DataPoint call(String line) {
		Vector<Integer> x_index=new Vector<Integer>();
		Vector<Double> x_value=new Vector<Double>();
		x_index.clear();
		x_value.clear();
		StringTokenizer itr = new StringTokenizer(line, " ");
		int line_num = Integer.parseInt(itr.nextToken());
		double y = Double.parseDouble(itr.nextToken());
		String tmp=itr.nextToken();
		while (tmp.contains(":")) {
			String[] strs=tmp.split(":");
			x_index.addElement(Integer.parseInt(strs[0]));
			x_value.addElement(Double.parseDouble(strs[1]));
			if (itr.hasMoreTokens()) tmp=itr.nextToken();
			else break;
		}
		return new DataPoint(line_num, new Xvector(x_index,x_value),y);
	}
}

class PrimalMap extends Function<DataPoint, DataPoint> {
	double[] weights;
	double[] p;
	int N;
	int T;
	double b;
	int r;
	
	public PrimalMap(double[] weights, double[] p, int N, int T, double b, int r) {
		this.weights = weights;
		this.p = p;
		this.N = N;
		this.T = T;
		this.b = b;
		this.r = r;
	}
	
	public double funcg(double tmp) {
		double res=0;
		res=1.0/(double)(1+Math.exp(tmp));
		return res;
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
	
	public DataPoint call(DataPoint point) {
		DataPoint gradient = new DataPoint(0, new Xvector(new Vector<Integer>(),new Vector<Double>()), 0);
		Random rnd = new Random();
		int line_num = point.line_num;
		//if (line_num!=100) return null;
		Xvector curx = point.x;
		double coef = 0;
		if (p[line_num-1] > (double)(r) / (double)(N))
			coef = point.y * funcg(point.y * (dot(weights, curx) + b));
		else return null;
		gradient.y = coef ;
		coef = coef * p[line_num-1];
		
		gradient.x.x_index.clear();
		gradient.x.x_value.clear();
		int num=curx.x_index.size();
		gradient.x.x_index=curx.x_index;
		for (int i = 0; i < num; i++)
			gradient.x.x_value.addElement(coef * curx.x_value.elementAt(i) /* / Math.sqrt(2 * T) */);
		
		return gradient;
	}
}

class PrimalReduce extends Function2<DataPoint, DataPoint, DataPoint> {
	public DataPoint call(DataPoint a, DataPoint b) {
		DataPoint result = new DataPoint(0, new Xvector(new Vector<Integer>(),new Vector<Double>()), 0);
		result.x.x_index.clear();
		result.x.x_value.clear();
		
		if (a==null) {
			if (b==null) return null;
			else {
				result.x.x_index=b.x.x_index;
				result.x.x_value=b.x.x_value;
				result.y=b.y;
				return result;
			}
		}
		if (b==null) {
			result.x.x_index=a.x.x_index;
			result.x.x_value=a.x.x_value;
			result.y=a.y;
			return result;
		}
		
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

class DualMap extends Function<DataPoint, Xvector> {
	public int D;
	public double weights[];
	public double b;	
	public int jt;
	public double eta;
	public double len;
	
	public DualMap(double[] weights, int D, int jt, double eta, double b, double len) {
		this.weights = weights;
		this.D = D;
		this.jt = jt;
		this.eta = eta;
		this.b = b;
		this.len = len;
	}
	
	public double clip(double a, double b) {
		return Math.max(Math.min(a,b),(-1)*b);
	}
	
	public Xvector call(DataPoint point) {
		Xvector mypair = new Xvector(new Vector<Integer>(),new Vector<Double>());
		mypair.x_index.clear();
		mypair.x_value.clear();

		double value = 0;
		int flag = point.x.x_index.indexOf(jt);
		if (flag != -1) value = point.x.x_value.elementAt(flag);
		double sigma = value * len / weights[jt] + b * point.y;
		double sigma_hat = sigma;//clip(sigma, 1.0/eta);
		double res = 1 - eta * sigma_hat + eta * sigma_hat * eta * sigma_hat;
		
		mypair.x_index.addElement(point.line_num);
		mypair.x_value.addElement(res);
		return mypair;
	}
}

class DualReduce extends Function2<Xvector, Xvector, Xvector> {
	public Xvector call(Xvector a, Xvector b) {
		Xvector result = new Xvector(new Vector<Integer>(),new Vector<Double>());
		result.x_index.clear();
		result.x_value.clear();
		for (int i=0;i<a.x_index.size();i++) {
			result.x_index.addElement(a.x_index.elementAt(i));
			result.x_value.addElement(a.x_value.elementAt(i));
		}
		for (int i=0;i<b.x_index.size();i++) {
			result.x_index.addElement(b.x_index.elementAt(i));
			result.x_value.addElement(b.x_value.elementAt(i));
		}
		return result;
	}
}

public class JavaHdfsLR {
	static JavaSparkContext sc=null;
	static String fname=null;
	static int N = 0;
	static int D = 0;
	static int T = 0;
	static double[] w = null;
	static double b = 0;
	static double[] p = null;
	static int jt;
	static double eta;
	static int r;

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
	
	public static void pInitial(String[] args) {
		if (args.length < 7) {
			System.err.println("Usage: JavaHdfsLR <master> <file> <N> <D> <iters> <eta> <r>");
			System.exit(1);
		}
		sc = new JavaSparkContext(args[0], "JavaHdfsLR", System.getenv("SPARK_HOME"), System.getenv("SPARK_EXAMPLES_JAR"));
		fname = args[1]; // hadoop_data (with line index)
		N = Integer.parseInt(args[2]);
		D = Integer.parseInt(args[3]);
		T = Integer.parseInt(args[4]);
		w = new double[D];
		for (int i = 0; i < D; i++)
			w[i] = 0;
		b = 0;
		p = new double[N];
		for (int i = 0; i < N; i++)
			p[i] = 1;
		double res = 0;
		for (int i = 0; i < N; i++)
			res = res + p[i] * p[i];
		res = Math.sqrt(res);
		for (int i = 0; i < N; i++)
			p[i] = p[i] / res;
		eta = Double.parseDouble(args[5]);
		r = Integer.parseInt(args[6]);;
		jt = 0;
	}

	public static int fSample() {
		int res=0;
		Random rnd=new Random();
		double r=rnd.nextDouble();
		double sum=0;
		double total=0;
		for (int i=0;i<D;i++)
			total=total+w[i]*w[i];
		for (int i=0;i<D;i++) {
			sum=sum+w[i]*w[i];
			if (r<sum/total) break;
			res=res+1;
		}
		return res-1;
	}
	
	public static void main(String[] args) throws Exception {
		// Parameter initialization
		pInitial(args);
		
		// load data for only one time and add to cache
		JavaRDD<String> lines = sc.textFile(fname);
		JavaRDD<DataPoint> points = lines.map(new ParsePoint()).cache();
		
		// Iterations
		for (int i = 1; i <= T; i++) {
			System.out.println("On iteration " + i);
			
			// Primal Part 
			DataPoint gradient = points.map(new PrimalMap(w, p, N, T, b, r)).reduce(new PrimalReduce());
			
			// w Update
			int num=gradient.x.x_index.size();
			for (int j = 0; j < num; j++) {
				int index=gradient.x.x_index.elementAt(j);
				w[index-1] += gradient.x.x_value.elementAt(j) * N;
			}
			b += gradient.y;
			
			// Sample in feature space
			jt = fSample();
			
			
			double len = 0;
			for (int j = 0; j < D; j++)
				len = len + w[j] * w[j];
			// Dual Part 
			Xvector pmod = points.map(new DualMap(w, D, jt, eta, b, len)).reduce(new DualReduce());
			// p Update
			num = pmod.x_index.size();
			if (num != N) System.out.println("Dual-Part Dimension Error!");
			for (int j = 0; j < num; j++) {
				int index = pmod.x_index.elementAt(j);
				double value = pmod.x_value.elementAt(j);
				p[index-1] *= value;
			}
			double res = 0;
			for (int j = 0; j < N; j++)
				res = res + p[j] * p[j];
			res = Math.sqrt(res);
			for (int j = 0; j < N; j++)
				p[j] = p[j] / res;
			
			System.out.println(w[0]);
			System.out.println(w[1]);
			System.out.println(b);
			System.out.println(jt);
		}

		System.out.println("All Iterations Completed!");
		printWeights(w,b);
		System.exit(0);
	}
}