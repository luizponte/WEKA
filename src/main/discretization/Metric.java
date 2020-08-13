package main.discretization;

public class Metric {
	
	public static double calculateInfoD(double numInstancesOfClass, double numTotalInstances) {
		double infoD = 0d;
		
		if(numTotalInstances > 0)
		{
			double x = (double) numInstancesOfClass / (double) numTotalInstances;
			double y = Math.log10(x) / Math.log10(2);
			infoD = (x * y) * (-1);
		}
		
		return infoD;
	}
	
	public static double calculateInfoAttribute(double[] numInstancesPerClass, double numTotalInstances) {
		double infoA = 0d;
		
		double numInstances = 0d;
		for(int i = 0; i < numInstancesPerClass.length; i++)
		{
			numInstances += numInstancesPerClass[i];
		}
		
		if((numInstances > 0) && (numTotalInstances > 0))
		{
			double x = (double) numInstances / (double) numTotalInstances;
			
			double y = 0d;
			for(int i = 0; i < numInstancesPerClass.length; i++)
			{
				double numInstancesClass = numInstancesPerClass[i];
				y += calculateInfoD(numInstancesClass, numInstances);
			}
			
			infoA = x * y;
		}
		
		return infoA;
	}
	
	public static double calculateGain(double infoD, double infoAttribute) {
		double gain = (double) infoD - (double) infoAttribute;
		return gain;
	}
	
	public static double calculateSplitInfo(double numInstancesInBin, double numTotalInstances) {
		double splitInfo = calculateInfoD(numInstancesInBin, numTotalInstances);
		return splitInfo;
	}
	
	public static double calculateGainRatio(double gain, double splitInfo) {
		double gainRatio = (double) gain / (double) splitInfo;
		return gainRatio;
	}
	
	public static double calculateGiniD(double[] numInstancesPerClass) {
		double giniD = 0d;
		
		double numInstances = 0d;
		for(int i = 0; i < numInstancesPerClass.length; i++)
		{
			numInstances += numInstancesPerClass[i];
		}
		
		if(numInstances > 0)
		{
			double x2 = 0d;
			for(int i = 0; i < numInstancesPerClass.length; i++)
			{
				double x = (double) numInstancesPerClass[i] / (double) numInstances;
				x2 -= Math.pow(x, 2);
			}
			
			giniD = (double) 1d + (double) x2;
		}
		
		return giniD;
	}
	
	public static double calculateGiniAttribute(double[] numInstancesPerClass, double numTotalInstances) {
		double giniA = 0d;
		
		double numInstances = 0d;
		for(int i = 0; i < numInstancesPerClass.length; i++)
		{
			numInstances += numInstancesPerClass[i];
		}
		
		if((numInstances > 0) && (numTotalInstances > 0))
		{
			double x = (double) numInstances / (double) numTotalInstances;
			double giniD = calculateGiniD(numInstancesPerClass);
			giniA = (double) x * giniD;
		}
		
		return giniA;
	}
	
	public static double calculateDeltaGini(double giniD, double giniAttribute) {
		double deltaGini = (double) giniD - (double) giniAttribute;
		return deltaGini;
	}

}
