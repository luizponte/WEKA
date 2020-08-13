package main.utilities;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class Utility {
	
	public static Instances loadData(String filename) throws Exception {
		DataSource source = new DataSource(filename);
		Instances instances = source.getDataSet();
		
		if(instances.classIndex() < 0)
		{
			instances.setClassIndex(instances.numAttributes() - 1);
		}
		
		return instances;
	}
	
	public static void saveData(Instances instances, String filename) throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		File output = new File(filename);
		saver.setFile(output);
		saver.writeBatch();
	}
	
	public static void printInstances(Instances instances) {
		for(int index = 0; index < instances.numInstances(); index++)
		{
			Instance instance = instances.instance(index);
			System.out.print("Instance\t" + index + "\t[");
			for(int attributeIndex = 0; attributeIndex < instance.numAttributes() - 1; attributeIndex++)
			{
				System.out.print(instance.toString(attributeIndex) + ",");
			}
			System.out.print(instance.toString(instance.numAttributes() - 1));
			System.out.println("]");
		}
		System.out.print("\n");
	}
	
	public static boolean isGreaterThanOrEqualTo(Double x, Double y) {
		if(Double.compare(x, y) >= 0)
		{
			return true;
		}
		return false;
	}
	
	public static Instances getEmptyDataset(Instances dataset) {
		Instances empty = new Instances(dataset);
		for(int index = dataset.numInstances() - 1; index >= 0; index--)
		{
			empty.remove(index);
		}
		return empty;
	}
	
	public static double roundDouble(double value) {
		BigDecimal bd = new BigDecimal(value);
		bd = bd.setScale(5, RoundingMode.HALF_UP);
		return bd.doubleValue();
	}
	
	public static double truncate(double value) {
		BigDecimal bd = new BigDecimal(value);
		bd = bd.setScale(5, RoundingMode.DOWN);
		return bd.doubleValue();
	}
	
	public static Instances addIdForInstances(Instances dataset) {
		Instances instances = new Instances(dataset);
		Attribute attribute = new Attribute("ID");
		instances.insertAttributeAt(attribute, 0);
		
		for(int index = 0; index < instances.numInstances(); index++)
		{
			Instance instance = instances.get(index);
			instance.setValue(0, index+1);
		}
		
		instances.setClassIndex(instances.numAttributes() - 1);
		return instances;
	}
	
	public static Instances removeIdOfInstances(Instances dataset) {
		Instances removed = new Instances(dataset);
		removed.deleteAttributeAt(0);
		removed.setClassIndex(removed.numAttributes() - 1);
		return removed;
	}
	
	public static Instances sortInstancesById(Instances dataset) {
		Map<Integer, Instance> instancesMap = new HashMap<>();
		
		for(int index = 0; index < dataset.numInstances(); index++)
		{
			Instance instance = dataset.get(index);
			int id = (int) instance.value(0);
			instancesMap.put(id, instance);
		}
		
		Instances sorted = getEmptyDataset(dataset);
		
		for(int i = 1; i <= dataset.numInstances(); i++)
		{
			Instance instance = instancesMap.get(i);
			sorted.add(instance);
		}
		
		return sorted;
	}
	
	public static List<Integer> getRandomNumbers(Random random, int size) {
		List<Integer> values = new ArrayList<>();
		
		while(values.size() < size)
		{
			// generates values between 0 and the specified maximal value
			Integer value = random.nextInt(size);
			if(!values.contains(value))
			{
				values.add(value);
			}
		}
		
		return values;
	}
	
	public static Map<Integer,List<Integer>> getRandomNumbersForEachAttribute(Random random, int size, int numAttributes) {
		Map<Integer,List<Integer>> randomValues = new HashMap<>();
		
		for(int run = 0; run < numAttributes; run++)
		{
			List<Integer> values = getRandomNumbers(random, size);
			randomValues.put(run, values);
		}
		
		return randomValues;
	}
	
	public static Map<Integer,List<Integer>> randomlySortEachAttribute(List<Integer> indexes, Map<Integer,List<Integer>> randomValues) {
		Map<Integer,List<Integer>> randomlySorted = new HashMap<>();
		
		// iterates over each attribute
		for(Entry<Integer, List<Integer>> entry : randomValues.entrySet())
		{
			// the index of attribute
			Integer attribute = entry.getKey();
			
			// the shuffled order of indexes
			List<Integer> values = entry.getValue();
			
			List<Integer> sorted = new ArrayList<>();
			
			for(int index = 0; index < values.size(); index++)
			{
				sorted.add(indexes.get(values.get(index)));
			}
			
			randomlySorted.put(attribute, sorted);
		}
		
		return randomlySorted;
	}
	
	public static boolean isRegression(Instances instances) {
		Attribute classAttribute = instances.classAttribute();
		return classAttribute.isNumeric() || classAttribute.isDate();
	}
	
	public static void write(String path, List<String> lines) {
		System.out.println("Writing file ...");
		File file = new File(path);
		FileWriter fw = null;
		
		try {
			fw = new FileWriter(file);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		for(int index = 0; index < lines.size(); index++)
		{
			String line = lines.get(index);
			try {
				fw.write(line);
				fw.write("\n");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		try {
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
