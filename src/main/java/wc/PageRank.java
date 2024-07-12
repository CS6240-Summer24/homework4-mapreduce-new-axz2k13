package wc;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.URI;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

public class PageRank extends Configured implements Tool {
	private static final Logger logger = LogManager.getLogger(PageRank.class);
    private static enum DANGLING_VALUE {
        VALUE
    }

	public static class TokenizerMapper extends Mapper<Object, Text, IntWritable, FloatWritable> {
        private final Map<Integer, Integer> cache = new HashMap<>(); //Integers are a lot smaller than Strings

        @Override
		public void setup(Context context) throws IOException, InterruptedException {
            URI[] cacheFiles = context.getCacheFiles();
            long MAX = -1;//context.getConfiguration().getLong("limit", Integer.MAX_VALUE);
            for (URI cacheFile : cacheFiles) {
                FileSystem fs = FileSystem.get(new Path("s3://cs6240-hw4-mr-bucket/").toUri(), context.getConfiguration());
                //FileSystem fs = FileSystem.get(context.getConfiguration()); 
                // Path getFilePath = new Path(cacheFile.toString()); 
                // BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open("s3a://cs6240-hw4-mr-bucket/input/graph.txt"))); 
                fs.setWorkingDirectory(new Path("/input/"));
                FSDataInputStream in = fs.open(new Path("graph.txt"));
                BufferedReader reader = new BufferedReader(new InputStreamReader(in));
                String line = reader.readLine();
                while (line != null) {
                    String[] words = line.split(",");
                    if (words.length == 3) {
                        int left = Integer.parseInt(words[0]);
                        int right = Integer.parseInt(words[1]);
                        if (MAX <= 0 || (left <= MAX && right <= MAX)) {
                            // here we assume every node has at most one outgoing edge
                            cache.put(left, right);
                        }
                    }
                    line = reader.readLine();
                }
                System.out.println("Mapper setup complete");
                reader.close();
            }
        }

        private final IntWritable keyOut = new IntWritable();
        private final FloatWritable valueOut = new FloatWritable();
        private final FloatWritable danglingValue = new FloatWritable();

		@Override
		public void map(final Object key, final Text value, final Context context) throws IOException, InterruptedException {
			final String[] words = value.toString().split(",");
			if (words.length == 2) {
                int left = Integer.parseInt(words[0]);
                int right = cache.get(left);
                float rank = Float.parseFloat(words[1]);
                if (right == 0) {
                    context.getCounter(DANGLING_VALUE.VALUE).increment((long) (rank*10000000));
                    danglingValue.set(danglingValue.get() + rank);
                } else {
                    keyOut.set(right);
                    valueOut.set(rank);
                    // here we assume every node only has one outgoing edge
                    context.write(keyOut, valueOut);
                    keyOut.set(left);
                    valueOut.set(0);
                    context.write(keyOut, valueOut);
                }
            }
		}
	}

	public static class FloatSumReducer extends Reducer<IntWritable, FloatWritable, IntWritable, FloatWritable> {
		private final FloatWritable result = new FloatWritable();

		@Override
		public void reduce(final IntWritable key, final Iterable<FloatWritable> values, final Context context) throws IOException, InterruptedException {
			float sum = 0;
			for (final FloatWritable val : values) {
				sum += val.get();
			}
            //int k = Integer.parseInt(context.getConfiguration().get("k"));
            //float danglingValue = Float.parseFloat(context.getConfiguration().get("danglingValue"));
			result.set(sum);
			context.write(key, result);
		}
	}

    public static class DanglingMapper extends Mapper<Object, Text, IntWritable, FloatWritable> {
        private final IntWritable keyOut = new IntWritable();
        private final FloatWritable valueOut = new FloatWritable();

		@Override
		public void map(final Object key, final Text value, final Context context) throws IOException, InterruptedException {
			final String[] words = value.toString().split(",");
			if (words.length == 2) {
                int left = Integer.parseInt(words[0]);
                float rank = Float.parseFloat(words[1]);
                if (left != 0) {
                    float danglingValue = Float.parseFloat(context.getConfiguration().get("danglingValue"));
                    keyOut.set(left);
                    valueOut.set(rank + danglingValue);
                    context.write(keyOut, valueOut);
                }
            }
		}
	}

	@Override
	public int run(final String[] args) throws Exception {
		final Configuration conf = getConf();
		final Job job = Job.getInstance(conf, "PageRank");
		job.setJarByClass(PageRank.class);
		final Configuration jobConf = job.getConfiguration();
		jobConf.set("mapreduce.output.textoutputformat.separator", ",");
        jobConf.set("k", args[2]);
        job.addCacheFile(new Path(args[0]+"/graph.txt").toUri());
		// Delete output directory, only to ease local development; will not work on AWS. ===========
//		final FileSystem fileSystem = FileSystem.get(conf);
//		if (fileSystem.exists(new Path(args[1]))) {
//			fileSystem.delete(new Path(args[1]), true);
//		}
		// ================
		job.setMapperClass(TokenizerMapper.class);
		//job.setCombinerClass(FloatSumReducer.class);
		job.setReducerClass(FloatSumReducer.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(FloatWritable.class);
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]+"/temp"));
		job.waitForCompletion(true);

        final Configuration conf2 = getConf();
		final Job job2 = Job.getInstance(conf2, "Resolve Dangling Page Rank");
        job2.setJarByClass(PageRank.class);
		final Configuration jobConf2 = job2.getConfiguration();
		jobConf2.set("mapreduce.output.textoutputformat.separator", ",");

        float dValue = (float)job.getCounters().findCounter(DANGLING_VALUE.VALUE).getValue()/10000000;
        int k = Integer.parseInt(args[2]);
        jobConf2.set("danglingValue", "" + (dValue/(k*k)));

        FileInputFormat.addInputPath(job2, new Path(args[1]+"/temp"));
		FileOutputFormat.setOutputPath(job2, new Path(args[1]+"/final"));
        job2.setMapperClass(DanglingMapper.class);
        job2.setOutputKeyClass(IntWritable.class);
		job2.setOutputValueClass(FloatWritable.class);
        return job2.waitForCompletion(true) ? 0 : 1;
	}

	public static void main(final String[] args) {
		if (args.length != 2) {
			throw new Error("Two arguments required:\n<input-dir> <output-dir>");
		}
        int k = 10000;
        int iterations = 10;
        // create input files
        // try {
        //     File file = new File("input/ranks.txt");
        //     file.getParentFile().mkdirs();

        //     PrintWriter printWriter = new PrintWriter(file);
        //     for(int i = 1; i < k*k + 1; i++) {
        //         printWriter.println(String.format("%d,%f", i, (i == 0) ? 0 : 1.0/(k*k)));
        //     }
        //     printWriter.close();

        //     File file2 = new File("input/graph.txt");
        //     file2.getParentFile().mkdirs();

        //     printWriter = new PrintWriter(file2);
        //     for(int i = 0; i < k; i ++) {
        //         for (int j = 0; j < k - 1; j++) {
        //             printWriter.println(String.format("%d,%d,edge", i * k + 1 + j, i * k + 2 + j));
        //         }
        //         printWriter.println(String.format("%d,%d,edge", (i + 1) * k, 0));
        //     }
        //     printWriter.close();
        // } catch (final Exception e) {
        //     logger.error("", e);
        // }
        
        String[] newArgs = Arrays.copyOf(args, args.length + 1);
        newArgs[newArgs.length - 1] = Integer.toString(k);
		try {
            newArgs[1] = args[1]+"/iter1";
            ToolRunner.run(new PageRank(), newArgs);
            for (int i = 1; i < iterations; i++) {
                newArgs[0] = args[1]+String.format("/iter%d/final", i);
                newArgs[1] = args[1]+String.format("/iter%d", i + 1);
                ToolRunner.run(new PageRank(), newArgs);
            }
		} catch (final Exception e) {
			logger.error("", e);
		}
	}

}