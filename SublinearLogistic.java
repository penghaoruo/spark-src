import java.io.*;
import java.util.*;

class Xvector {
	Vector<Integer> x_index;
	Vector<Double> x_value;
	public Xvector(Vector<Integer> x_index, Vector<Double> x_value) {
		this.x_index = x_index;
		this.x_value = x_value;
	}
}

class DataPoint {
	Xvector x;
	double y;
	public DataPoint(Xvector x, double y) {
		this.x = x;
		this.y = y;
	}
}

public class SublinearLogistic {
	static String fname=null;
	static int N = 0;
	static int D = 0;
	static int T = 0;
	static double[] w = null;
	static double b = 0;
	static double[] p = null;
	static int it;
	static int jt;
	static double eta;
	static DataPoint[] points = null;
	static double b_avg = 0;

	public static double funcg(double tmp) {
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
	
	public static void primal() {
		Xvector curx = points[jt].x;
		double y = points[jt].y;
		double coef = y * funcg(y * (dot(w, curx) + b)) / Math.sqrt(2 * T);
		int num=curx.x_index.size();
		for (int i = 0; i < num; i++)
			w[curx.x_index.elementAt(i)-1] += coef * curx.x_value.elementAt(i) ;
		double res = 0;
		for (int j = 0; j < D; j++)
			res = res + w[j] * w[j];
		res = Math.sqrt(res);
		for (int j = 0; j < D; j++)
			w[j] = w[j] / res;
		
		res=0;
		for (int i=0;i<N;i++)
			res+=p[i]*points[i].y;
		if (res>0) b=1;
		if (res<0) b=-1;
		if (res==0) b=0;
	}
	
	public static double clip(double a, double b) {
		return Math.max(Math.min(a,b),(-1)*b);
	}
	
	public static void dual() {
		for (int i=0;i<N;i++) {
			double value = 0;
			int flag = points[i].x.x_index.indexOf(jt);
			if (flag != -1) value = points[i].x.x_value.elementAt(flag);
			double sigma = value / w[jt] + b * points[i].y;
			double sigma_hat = clip(sigma, 1.0/eta);
			p[i] *= 1 - eta * sigma_hat + eta * sigma_hat * eta * sigma_hat;
			double res = 0;
			for (int j = 0; j < N; j++)
				res = res + p[j] * p[j];
			res = Math.sqrt(res);
			for (int j = 0; j < N; j++)
				p[j] = p[j] / res;
		}
	}
	
	public static void ParsePoint() throws Exception {
		File f=new File(fname);
		FileReader reader = new FileReader(f);
		BufferedReader buf= new BufferedReader(reader);
		
		String line=null;
		line=buf.readLine();
		int index=0;
		while (line!=null) {
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
			points[index]=new DataPoint(new Xvector(x_index,x_value),y);
			line=buf.readLine();
			index++;
		}
		buf.close();
		reader.close();
	}

	public static void printWeights(double[] a, double b) throws Exception {
		File fout=new File("output-model.txt");
		FileWriter writer = new FileWriter(fout);
		BufferedWriter bw= new BufferedWriter(writer);
		for (int i=0;i<D;i++) {
			bw.write(a[i]+"\n");
			bw.flush();
		}
		bw.write(b_avg+"\n");
		bw.flush();
		bw.close();
		writer.close();
	}
	
	public static void pInitial(String[] args) {
		if (args.length < 4) {
			System.err.println("Usage: SublinearLogistic <file> <N> <D> <iters>");
			System.exit(1);
		}
		fname = args[0];
		N = Integer.parseInt(args[1]);
		D = Integer.parseInt(args[2]);
		T = Integer.parseInt(args[3]);
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
		points = new DataPoint[N];
		eta=0.15;
		
	}
	
	public static int iSample() {
		int res=0;
		Random rnd=new Random();
		double r=rnd.nextDouble();
		double sum=0;
		for (int i=0;i<N;i++) {
			sum=sum+p[i]*p[i];
			if (r<sum) break;
			res=res+1;
		}
		return res;
	}

	public static int fSample() {
		int res=0;
		Random rnd=new Random();
		double r=rnd.nextDouble();
		double sum=0;
		for (int i=0;i<D;i++) {
			sum=sum+w[i]*w[i];
			if (r<sum) break;
			res=res+1;
		}
		return res;
	}
	
	public static void main(String[] args) throws Exception {
		pInitial(args);
		ParsePoint();
		for (int i = 1; i <= T; i++) {
			System.out.println("On iteration " + i);
			it = iSample();
			primal();
			b_avg=(b_avg*(i-1)+b)/i;
			jt = fSample();
			dual();
		}

		System.out.println("All Iterations Completed!");
		printWeights(w,b);
		System.exit(0);
	}
}