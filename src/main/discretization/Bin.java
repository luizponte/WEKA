package main.discretization;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import main.utilities.Utility;
import weka.core.Instance;

public class Bin {
	
	private List<Instance> instances;
	private boolean singleClass;
	private double classValue;
	private double start;
	private double end;
	private double minimumValue;
	private double maximumValue;
	private int numberOfInstances;
	private double numberOfClasses;
	private double[] numberOfInstancesPerClass;
	private int attributeIndex;
	private double entropy;
	
	public Bin(List<Instance> instances, int attributeIndex, double start, double end) {
		this.instances = instances;
		numberOfInstances = this.instances.size();
		this.attributeIndex = attributeIndex;
		this.start = Utility.truncate(start);
		this.end = Utility.truncate(end);
		
		if(this.numberOfInstances > 0)
		{
			checkClasses(this.instances);
			setClassValue();
			calculateMinimumAndMaximum(this.instances);
			calculateEntropy();
			setNumberOfInstancesPerClass();
		}
		else
		{
			this.singleClass = true;
			this.classValue = Double.NaN;
			this.minimumValue = Double.NaN;
			this.maximumValue = Double.NaN;
			this.entropy = Double.NaN;
		}
	}
	
	public List<Instance> getInstances() {
		return this.instances;
	}
	
	public boolean isSingleClass() {
		return this.singleClass;
	}
	
	public double getClassValue() {
		return this.classValue;
	}
	
	public double getStart() {
		return this.start;
	}
	
	public double getEnd() {
		return this.end;
	}
	
	public double getMinimumValue() {
		return this.minimumValue;
	}
	
	public double getMaximumValue() {
		return this.maximumValue;
	}
	
	public int getNumberOfInstances() {
		return this.numberOfInstances;
	}
	
	public int getAttributeIndex() {
		return this.attributeIndex;
	}
	
	public double getEntropy() {
		return this.entropy;
	}
	
	public double getQuantityOfClasses() {
		return this.numberOfClasses;
	}
	
	public double[] getQuantityOfInstancesPerClass() {
		return this.numberOfInstancesPerClass;
	}
	
	public void setStart(double newStart) {
		this.start = Utility.truncate(newStart);
	}
	
	public void setEnd(double newEnd) {
		this.end = Utility.truncate(newEnd);
	}
	
	private void checkClasses(List<Instance> instances) {
		this.singleClass = true;
		double classValue = 0d;
		for(int index = 0; index < instances.size(); index++)
		{
			Instance instance = instances.get(index);
			if(index == 0)
			{
				classValue = instance.classValue();
			}
			else
			{
				if(Double.compare(classValue, instance.classValue()) != 0)
				{
					this.singleClass = false;
				}
			}
		}
	}
	
	private void setClassValue() {
		if(this.singleClass)
		{
			this.classValue = this.instances.get(0).classValue();
		}
		else
		{
			this.classValue = Double.NaN;
		}
	}
	
	private void calculateMinimumAndMaximum(List<Instance> instances) {
		double minimum = Double.POSITIVE_INFINITY, maximum = Double.NEGATIVE_INFINITY;
		
		for(int index = 0; index < instances.size(); index++)
		{
			Instance instance = instances.get(index);
			
			if(Double.compare(minimum, Utility.truncate(instance.value(this.attributeIndex))) > 0)
			{
				minimum = instance.value(this.attributeIndex);
			}
			
			if(Double.compare(maximum, Utility.truncate(instance.value(this.attributeIndex))) < 0)
			{
				maximum = instance.value(this.attributeIndex);
			}
			
		}
		
		this.minimumValue = Utility.truncate(minimum);
		this.maximumValue = Utility.truncate(maximum);
	}
	
	private void calculateEntropy() {
		double entropy = 0d;
		
		Map<Double, Integer> instancesPerClass = getNumberOfInstancesPerClass();
		Set<Double> classes = instancesPerClass.keySet();
		
		for(Double c : classes)
		{
			double quantityPerClass = (double) instancesPerClass.get(c);
			if(this.numberOfInstances == 0)
			{
				entropy += 0d;
			}
			else
			{
				double x = (double) quantityPerClass / (double) this.numberOfInstances;
				double y = Math.log10(x) / Math.log10(2);
				entropy += (x * y);
			}
			
		}
		
		if(entropy != 0)
		{
			entropy *= -1;
		}
		
		this.entropy = entropy;
	}
	
	private Map<Double, Integer> getNumberOfInstancesPerClass() {
		Map<Double, Integer> instancesPerClass = new HashMap<>();
		
		for(int index = 0; index < this.numberOfInstances; index++)
		{
			Instance instance = this.instances.get(index);
			double cValue = instance.classValue();
			if(instancesPerClass.containsKey(cValue))
			{
				int quantity = instancesPerClass.get(cValue);
				instancesPerClass.remove(cValue);
				instancesPerClass.put(cValue, quantity + 1);
			}
			else
			{
				instancesPerClass.put(cValue, 1);
			}
		}
		
		return instancesPerClass;
	}
	
	private void setNumberOfInstancesPerClass() {
		Map<Double, Integer> instancesPerClass = getNumberOfInstancesPerClass();
		Set<Double> classes = instancesPerClass.keySet();
		this.numberOfClasses = (double) classes.size();
		
		double[] numberOfInstancesPerClass = new double[classes.size()];
		
		int index = 0;
		for(Double key : classes)
		{
			Integer numberOfInstances = instancesPerClass.get(key);
			numberOfInstancesPerClass[index] = (double) numberOfInstances;
			index++;
		}
		
		this.numberOfInstancesPerClass = numberOfInstancesPerClass;
	}

}
