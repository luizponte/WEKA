package test.discretization;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import main.discretization.BruteForce;
import main.utilities.Utility;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class TestDiscretization {
	
	// change these values if needed
	public static int numberOfBins = 3;
	public static int crossValidationFolds = 5;
		
	// change these values if needed
	public static String inputTrain = "train";
	public static String inputTest = "test";
	public static String output = "_discretized_";
	public static String extension = ".arff";

	public static void main(String[] args) throws Exception {		
		train();
		test();
	}
		
	public static void train() throws Exception {
		for(int fold = 1; fold <= crossValidationFolds; fold++)
		{
			Instances dataset = Utility.loadData(inputTrain + fold + extension);
			dataset.setClassIndex(dataset.numAttributes() -1);
			
			if(dataset.classAttribute().isNominal())
			{
				List<Integer> attributes = new ArrayList<>();
				
				// add or remove here the attributes indexes to discretize
				attributes.add(0);
				attributes.add(1);
				attributes.add(2);
				attributes.add(3);
				attributes.add(4);
				attributes.add(5);
				attributes.add(6);
				attributes.add(7);
				attributes.add(8);
				attributes.add(9);
					
				Instances discretize = BruteForce.discretize(dataset, attributes, numberOfBins);
					
				Utility.saveData(discretize, inputTrain + fold + output + numberOfBins + extension);
				System.out.println("Supervised discretization using " + numberOfBins + " bins is done!");
			}
			else
			{
				System.out.println("Can not perform the supervised discretization for dataset without nominal class!");
			}
		}
	}
		
	public static void test() throws Exception {
		for(int fold = 1; fold <= crossValidationFolds; fold++)
		{
			Instances train = Utility.loadData(inputTrain + fold + output + numberOfBins + extension);
			train.setClassIndex(train.numAttributes() -1);
				
			Instances test = Utility.loadData(inputTest + fold + extension);
				
			Instances discretized = discretizeTestPartition(train, test);
					
			Utility.saveData(discretized, inputTest + fold + output + numberOfBins + extension);
			System.out.println("Test partition with " + numberOfBins + " bins was built!");
		}
	}
		
	private static Instances discretizeTestPartition(Instances train, Instances test) {
		Instances discretized = new Instances(test);
		Instances aux = removeAllInstances(discretized);
			
		for(int index = 0; index < train.numAttributes(); index++)
		{
			Attribute attribute = train.attribute(index);
				
			if(attribute.isNominal() && discretized.attribute(index).isNumeric())
			{
				aux.replaceAttributeAt(attribute, index);
					
				for(int i = 0; i < discretized.numInstances(); i++)
				{
					Instance instance = discretized.instance(i);
					Double instanceValue = instance.value(index);
						
					Enumeration<Object> values = attribute.enumerateValues();
						
					int count = 1;
						
					while(values.hasMoreElements())
					{
						String value = (String) values.nextElement();
						String[] range = value.split("/");
							
						Double min = Double.valueOf(range[0]);
						Double max = Double.valueOf(range[1]);
							
						if(count == 1 && max.compareTo(instanceValue) > 0)
						{
							instance.setDataset(aux);
							instance.setValue(index, value);
							aux.add(instance);
							break;
						}
						else if(count == attribute.numValues() && min.compareTo(instanceValue) <= 0)
						{
							instance.setDataset(aux);
							instance.setValue(index, value);
							aux.add(instance);
							break;
						}
						else if(min.compareTo(instanceValue) <= 0 && max.compareTo(instanceValue) > 0)
						{
							instance.setDataset(aux);
							instance.setValue(index, value);
							aux.add(instance);
							break;
						}
						count++;
					}
				}
				discretized = new Instances(aux);
				aux = removeAllInstances(discretized);
			}
		}
		
		return discretized;
	}
		
	private static Instances removeAllInstances(Instances dataset) {
		Instances empty = new Instances(dataset);
		for(int index = empty.numInstances() - 1; index >= 0; index--)
		{
			empty.remove(index);
		}
		return empty;
	}

}
