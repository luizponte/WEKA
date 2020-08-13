package main.discretization;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import main.utilities.Utility;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class BruteForce {
	
	public static boolean useGainRatio = false;
	public static boolean useGiniIndex = false;
	public static boolean useInfoGain = false;
	
	public static Instances discretize(Instances dataset, List<Integer> attributes, int numberOfBins) {
		Instances discretized = new Instances(dataset);
		
		if(attributes.size() > 0 && dataset.classAttribute().isNominal())
		{
			discretized = Utility.addIdForInstances(discretized);
			for(int element = 0; element < attributes.size(); element++)
			{
				int attributeIndex = attributes.get(element) + 1;
				
				// checks if the attribute index is valid and also if it is not equals to the class index
				if(attributeIndex < (discretized.numAttributes() - 1) && attributeIndex > 0)
				{
					double minimum = getMinimumAttributeValue(discretized, attributeIndex);
					BigDecimal min = new BigDecimal(String.valueOf(minimum));
					
					double maximum = getMaximumAttributeValue(discretized, attributeIndex);
					BigDecimal max = new BigDecimal(String.valueOf(maximum));
					
					if(!Double.isNaN(minimum) && !Double.isNaN(maximum))
					{
						BigDecimal numInstances = new BigDecimal(String.valueOf(discretized.numInstances()));
						BigDecimal maxMin = max.subtract(min);
						BigDecimal rangeSize = maxMin.divide(numInstances, maxMin.scale(), RoundingMode.DOWN);
						
						if(maxMin.scale() <= 1)
						{
							rangeSize = maxMin.divide(numInstances, 5, RoundingMode.DOWN);
						}
						
						// the number of the range and its values with all decimal places (does not round or truncate)
						// Map<0 to 54, [start,end]> ranges
						Map<Integer, Double[]> ranges = getRanges(min, max, rangeSize, discretized.numInstances());
						
						List<Bin> histogram = getHistogram(ranges, discretized, attributeIndex);
						
						List<Bin> bins = null;
						if(numberOfBins < 2 || numberOfBins > 5)
						{
							System.out.println("Error! The informed number of bins is not valid!");
						}
						else
						{
							bins = bruteForce(histogram, numberOfBins, rangeSize.doubleValue());
							discretized = buildDataset(bins, discretized);
							System.out.println("Done with " + getNumberOfBinInstances(histogram) + " instances!");
						}
					}
					else
					{
						System.out.println("Error! This attribute has only missing values!");
					}
					
				}
				else
				{
					System.out.println("Invalid attribute index (" + (attributeIndex - 1) + ")!");
				}
			}
			
			discretized = Utility.sortInstancesById(discretized);
			discretized = Utility.removeIdOfInstances(discretized);
			discretized.setRelationName(dataset.relationName() + "_discretized");
		}
		else
		{
			System.out.println("This dataset's class is not nominal. The discretization can not be done!");
		}
		
		return discretized;

	}
	
	private static List<Bin> bruteForce(List<Bin> bins, int numberOfBins, double rangeSize) {
		
		List<Bin> grouped = removeEmptyBins(bins, rangeSize);
		
		if(grouped.size() <= numberOfBins)
		{
			return grouped;
		}
		
		if(numberOfBins == 2)
		{
			return groupBins2(grouped, numberOfBins);
		}
		else
		{
			if(numberOfBins == 3)
			{
				return groupBins3(grouped, numberOfBins);
			}
			else
			{
				if(numberOfBins == 4)
				{
					return groupBins4(grouped, numberOfBins);
				}
				else
				{
					return groupBins5(grouped, numberOfBins);
				}
			}
		}
		
	}
	
	private static Bin getPartition(List<Bin> bins, int start, int end) {
		Bin partition = bins.get(start);
		
		for(int index = (start + 1); index <= end; index++)
		{
			Bin bin = bins.get(index);
			partition = groupBins(partition, bin);
		}
		
		return partition;
	}
	
	private static List<Bin> groupBins2(List<Bin> bins, int numberOfBins) {
		List<Bin> grouped = new ArrayList<>(bins);
		List<Bin> aux = new ArrayList<>();
		
		Double entropy = Double.POSITIVE_INFINITY;
		Double gainRatio = Double.NEGATIVE_INFINITY;
		Double deltaGini = Double.NEGATIVE_INFINITY;
		Double infoGain = Double.NEGATIVE_INFINITY;
		
		for(int i = 0; i < grouped.size() - 1; i++)
		{
			Bin partition1 = getPartition(bins, 0, i);
			Bin partition2 = getPartition(bins, (i + 1), (grouped.size() - 1));
			
			if(useGainRatio)
			{
				double weightedMean = calculateGainRatioFor2Bins(partition1, partition2);
				
				if(Double.compare(gainRatio, weightedMean) < 0)
				{
					gainRatio = weightedMean;
				}
			}
			else
			{
				if(useGiniIndex)
				{
					double weightedMean = calculateDeltaGiniIndexFor2Bins(partition1, partition2);
					
					if(Double.compare(deltaGini, weightedMean) < 0)
					{
						deltaGini = weightedMean;
					}
				}
				else
				{
					if(useInfoGain)
					{
						double weightedMean = calculateInfoGainFor2Bins(partition1, partition2);
						
						if(Double.compare(infoGain, weightedMean) < 0)
						{
							infoGain = weightedMean;
						}
					}
					else
					{
						double weightedMean = calculateEntropyFor2Bins(partition1, partition2);
						
						if(Double.compare(entropy, weightedMean) > 0)
						{
							entropy = weightedMean;
						}
					}
					
				}
			}
			
		}
		
		boolean stop = false;
		for(int i = 0; i < grouped.size() - 1; i++)
		{
			Bin partition1 = getPartition(bins, 0, i);
			Bin partition2 = getPartition(bins, (i + 1), (grouped.size() - 1));
			
			if(useGainRatio)
			{
				double weightedMean = calculateGainRatioFor2Bins(partition1, partition2);
				
				if(!stop && (Double.compare(gainRatio, weightedMean) == 0))
				{
					aux.add(partition1);
					aux.add(partition2);
					
					stop = true;
					break;
				}
			}
			else
			{
				if(useGiniIndex)
				{
					double weightedMean = calculateDeltaGiniIndexFor2Bins(partition1, partition2);
					
					if(!stop && (Double.compare(deltaGini, weightedMean) == 0))
					{
						aux.add(partition1);
						aux.add(partition2);
						
						stop = true;
						break;
					}
				}
				else
				{
					if(useInfoGain)
					{
						double weightedMean = calculateInfoGainFor2Bins(partition1, partition2);
						
						if(!stop && (Double.compare(infoGain, weightedMean) == 0))
						{
							aux.add(partition1);
							aux.add(partition2);
							
							stop = true;
							break;
						}
					}
					else
					{
						double weightedMean = calculateEntropyFor2Bins(partition1, partition2);
						
						if(!stop && (Double.compare(entropy, weightedMean) == 0))
						{
							aux.add(partition1);
							aux.add(partition2);
							
							stop = true;
							break;
						}
					}
				}
			}
			
		}
		
		String line = "Entropy;" + entropy + ";InfoGain;" + infoGain + ";Gain Ratio;" + gainRatio + ";Delta Gini Index;" + deltaGini + ";";
		System.out.println(line);
		
		return aux;
	}
	
	private static List<Bin> groupBins3(List<Bin> bins, int numberOfBins) {
		List<Bin> grouped = new ArrayList<>(bins);
		List<Bin> aux = new ArrayList<>();
		
		Double entropy = Double.POSITIVE_INFINITY;
		Double gainRatio = Double.NEGATIVE_INFINITY;
		Double deltaGini = Double.NEGATIVE_INFINITY;
		Double infoGain = Double.NEGATIVE_INFINITY;
		
		for(int i = 0; i < grouped.size() - 2; i++)
		{
			for(int j = (i + 1); j < grouped.size() - 1; j++)
			{
				Bin partition1 = getPartition(bins, 0, i);
				Bin partition2 = getPartition(bins, (i + 1), j);
				Bin partition3 = getPartition(bins, (j + 1), (grouped.size() - 1));
				
				if(useGainRatio)
				{
					double weightedMean = calculateGainRatioFor3Bins(partition1, partition2, partition3);
					
					if(Double.compare(gainRatio, weightedMean) < 0)
					{
						gainRatio = weightedMean;
					}
				}
				else
				{
					if(useGiniIndex)
					{
						double weightedMean = calculateDeltaGiniIndexFor3Bins(partition1, partition2, partition3);
						
						if(Double.compare(deltaGini, weightedMean) < 0)
						{
							deltaGini = weightedMean;
						}
					}
					else
					{
						if(useInfoGain)
						{
							double weightedMean = calculateInfoGainFor3Bins(partition1, partition2, partition3);
							
							if(Double.compare(infoGain, weightedMean) < 0)
							{
								infoGain = weightedMean;
							}
						}
						else
						{
							double weightedMean = calculateEntropyFor3Bins(partition1, partition2, partition3);
							
							if(Double.compare(entropy, weightedMean) > 0)
							{
								entropy = weightedMean;
							}
						}
					}
				}
				
			}
		}
		
		boolean stop = false;
		for(int i = 0; i < grouped.size() - 2; i++)
		{
			for(int j = (i + 1); j < grouped.size() - 1; j++)
			{
				Bin partition1 = getPartition(bins, 0, i);
				Bin partition2 = getPartition(bins, (i + 1), j);
				Bin partition3 = getPartition(bins, (j + 1), (grouped.size() - 1));
				
				if(useGainRatio)
				{
					double weightedMean = calculateGainRatioFor3Bins(partition1, partition2, partition3);
					
					if(!stop && (Double.compare(gainRatio, weightedMean) == 0))
					{
						aux.add(partition1);
						aux.add(partition2);
						aux.add(partition3);
						
						stop = true;
						break;
					}
				}
				else
				{
					if(useGiniIndex)
					{
						double weightedMean = calculateDeltaGiniIndexFor3Bins(partition1, partition2, partition3);
						
						if(!stop && (Double.compare(deltaGini, weightedMean) == 0))
						{
							aux.add(partition1);
							aux.add(partition2);
							aux.add(partition3);
							
							stop = true;
							break;
						}
					}
					else
					{
						if(useInfoGain)
						{
							double weightedMean = calculateInfoGainFor3Bins(partition1, partition2, partition3);
							
							if(!stop && (Double.compare(infoGain, weightedMean) == 0))
							{
								aux.add(partition1);
								aux.add(partition2);
								aux.add(partition3);
								
								stop = true;
								break;
							}
						}
						else
						{
							double weightedMean = calculateEntropyFor3Bins(partition1, partition2, partition3);
							
							if(!stop && (Double.compare(entropy, weightedMean) == 0))
							{
								aux.add(partition1);
								aux.add(partition2);
								aux.add(partition3);
								
								stop = true;
								break;
							}
						}
					}
				}
				
			}
		}
		
		String line = "Entropy;" + entropy + ";InfoGain;" + infoGain + ";Gain Ratio;" + gainRatio + ";Delta Gini Index;" + deltaGini + ";";
		System.out.println(line);
		
		return aux;
	}
	
	private static List<Bin> groupBins4(List<Bin> bins, int numberOfBins) {
		List<Bin> grouped = new ArrayList<>(bins);
		List<Bin> aux = new ArrayList<>();
		
		Double entropy = Double.POSITIVE_INFINITY;
		Double gainRatio = Double.NEGATIVE_INFINITY;
		Double deltaGini = Double.NEGATIVE_INFINITY;
		Double infoGain = Double.NEGATIVE_INFINITY;
		
		for(int i = 0; i < grouped.size() - 3; i++)
		{
			for(int j = (i + 1); j < grouped.size() - 2; j++)
			{
				for(int k = (j + 1); k < grouped.size() - 1; k++)
				{
					Bin partition1 = getPartition(bins, 0, i);
					Bin partition2 = getPartition(bins, (i + 1), j);
					Bin partition3 = getPartition(bins, (j + 1), k);
					Bin partition4 = getPartition(bins, (k + 1), (grouped.size() - 1));
					
					if(useGainRatio)
					{
						double weightedMean = calculateGainRatioFor4Bins(partition1, partition2, partition3, partition4);
						
						if(Double.compare(gainRatio, weightedMean) < 0)
						{
							gainRatio = weightedMean;
						}
					}
					else
					{
						if(useGiniIndex)
						{
							double weightedMean = calculateDeltaGiniIndexFor4Bins(partition1, partition2, partition3, partition4);
							
							if(Double.compare(deltaGini, weightedMean) < 0)
							{
								deltaGini = weightedMean;
							}
						}
						else
						{
							if(useInfoGain)
							{
								double weightedMean = calculateInfoGainFor4Bins(partition1, partition2, partition3, partition4);
								
								if(Double.compare(infoGain, weightedMean) < 0)
								{
									infoGain = weightedMean;
								}
							}
							else
							{
								double weightedMean = calculateEntropyFor4Bins(partition1, partition2, partition3, partition4);
								
								if(Double.compare(entropy, weightedMean) > 0)
								{
									entropy = weightedMean;
								}
							}
						}
					}
					
				}
			}
		}
		
		boolean stop = false;
		for(int i = 0; i < grouped.size() - 3; i++)
		{
			for(int j = (i + 1); j < grouped.size() - 2; j++)
			{
				for(int k = (j + 1); k < grouped.size() - 1; k++)
				{
					Bin partition1 = getPartition(bins, 0, i);
					Bin partition2 = getPartition(bins, (i + 1), j);
					Bin partition3 = getPartition(bins, (j + 1), k);
					Bin partition4 = getPartition(bins, (k + 1), (grouped.size() - 1));
					
					if(useGainRatio)
					{
						double weightedMean = calculateGainRatioFor4Bins(partition1, partition2, partition3, partition4);
						
						if(!stop && (Double.compare(gainRatio, weightedMean) == 0))
						{
							aux.add(partition1);
							aux.add(partition2);
							aux.add(partition3);
							aux.add(partition4);
							
							stop = true;
							break;
						}
					}
					else
					{
						if(useGiniIndex)
						{
							double weightedMean = calculateDeltaGiniIndexFor4Bins(partition1, partition2, partition3, partition4);
							
							if(!stop && (Double.compare(deltaGini, weightedMean) == 0))
							{
								aux.add(partition1);
								aux.add(partition2);
								aux.add(partition3);
								aux.add(partition4);
								
								stop = true;
								break;
							}
						}
						else
						{
							if(useInfoGain)
							{
								double weightedMean = calculateInfoGainFor4Bins(partition1, partition2, partition3, partition4);
								
								if(!stop && (Double.compare(infoGain, weightedMean) == 0))
								{
									aux.add(partition1);
									aux.add(partition2);
									aux.add(partition3);
									aux.add(partition4);
									
									stop = true;
									break;
								}
							}
							else
							{
								double weightedMean = calculateEntropyFor4Bins(partition1, partition2, partition3, partition4);
								
								if(!stop && (Double.compare(entropy, weightedMean) == 0))
								{
									aux.add(partition1);
									aux.add(partition2);
									aux.add(partition3);
									aux.add(partition4);
									
									stop = true;
									break;
								}
							}
						}
					}
					
				}
			}
		}
		
		String line = "Entropy;" + entropy + ";InfoGain;" + infoGain + ";Gain Ratio;" + gainRatio + ";Delta Gini Index;" + deltaGini + ";";
		System.out.println(line);
		
		return aux;
	}
	
	private static List<Bin> groupBins5(List<Bin> bins, int numberOfBins) {
		List<Bin> grouped = new ArrayList<>(bins);
		List<Bin> aux = new ArrayList<>();
		
		Double entropy = Double.POSITIVE_INFINITY;
		Double gainRatio = Double.NEGATIVE_INFINITY;
		Double deltaGini = Double.NEGATIVE_INFINITY;
		Double infoGain = Double.NEGATIVE_INFINITY;
		
		for(int i = 0; i < grouped.size() - 4; i++)
		{
			for(int j = (i + 1); j < grouped.size() - 3; j++)
			{
				for(int k = (j + 1); k < grouped.size() - 2; k++)
				{
					for(int m = (k + 1); m < grouped.size() - 1; m++)
					{
						Bin partition1 = getPartition(bins, 0, i);
						Bin partition2 = getPartition(bins, (i + 1), j);
						Bin partition3 = getPartition(bins, (j + 1), k);
						Bin partition4 = getPartition(bins, (k + 1), m);
						Bin partition5 = getPartition(bins, (m + 1), (grouped.size() - 1));
						
						if(useGainRatio)
						{
							double weightedMean = calculateGainRatioFor5Bins(partition1, partition2, partition3, partition4, partition5);
							
							if(Double.compare(gainRatio, weightedMean) < 0)
							{
								gainRatio = weightedMean;
							}
						}
						else
						{
							if(useGiniIndex)
							{
								double weightedMean = calculateDeltaGiniIndexFor5Bins(partition1, partition2, partition3, partition4, partition5);
								
								if(Double.compare(deltaGini, weightedMean) < 0)
								{
									deltaGini = weightedMean;
								}
							}
							else
							{
								if(useInfoGain)
								{
									double weightedMean = calculateInfoGainFor5Bins(partition1, partition2, partition3, partition4, partition5);
									
									if(Double.compare(infoGain, weightedMean) < 0)
									{
										infoGain = weightedMean;
									}
								}
								else
								{
									double weightedMean = calculateEntropyFor5Bins(partition1, partition2, partition3, partition4, partition5);
									
									if(Double.compare(entropy, weightedMean) > 0)
									{
										entropy = weightedMean;
									}
								}
							}
						}
						
					}
				}
			}
		}
		
		boolean stop = false;
		for(int i = 0; i < grouped.size() - 4; i++)
		{
			for(int j = (i + 1); j < grouped.size() - 3; j++)
			{
				for(int k = (j + 1); k < grouped.size() - 2; k++)
				{
					for(int m = (k + 1); m < grouped.size() - 1; m++)
					{
						Bin partition1 = getPartition(bins, 0, i);
						Bin partition2 = getPartition(bins, (i + 1), j);
						Bin partition3 = getPartition(bins, (j + 1), k);
						Bin partition4 = getPartition(bins, (k + 1), m);
						Bin partition5 = getPartition(bins, (m + 1), (grouped.size() - 1));
						
						if(useGainRatio)
						{
							double weightedMean = calculateGainRatioFor5Bins(partition1, partition2, partition3, partition4, partition5);
							
							if(!stop && (Double.compare(gainRatio, weightedMean) == 0))
							{
								aux.add(partition1);
								aux.add(partition2);
								aux.add(partition3);
								aux.add(partition4);
								aux.add(partition5);
								
								stop = true;
								break;
							}
						}
						else
						{
							if(useGiniIndex)
							{
								double weightedMean = calculateDeltaGiniIndexFor5Bins(partition1, partition2, partition3, partition4, partition5);
								
								if(!stop && (Double.compare(deltaGini, weightedMean) == 0))
								{
									aux.add(partition1);
									aux.add(partition2);
									aux.add(partition3);
									aux.add(partition4);
									aux.add(partition5);
									
									stop = true;
									break;
								}
							}
							else
							{
								if(useInfoGain)
								{
									double weightedMean = calculateInfoGainFor5Bins(partition1, partition2, partition3, partition4, partition5);
									
									if(!stop && (Double.compare(infoGain, weightedMean) == 0))
									{
										aux.add(partition1);
										aux.add(partition2);
										aux.add(partition3);
										aux.add(partition4);
										aux.add(partition5);
										
										stop = true;
										break;
									}
								}
								else
								{
									double weightedMean = calculateEntropyFor5Bins(partition1, partition2, partition3, partition4, partition5);
									
									if(!stop && (Double.compare(entropy, weightedMean) == 0))
									{
										aux.add(partition1);
										aux.add(partition2);
										aux.add(partition3);
										aux.add(partition4);
										aux.add(partition5);
										
										stop = true;
										break;
									}
								}
							}
						}
						
					}
				}
			}
		}
		
		String line = "Entropy;" + entropy + ";InfoGain;" + infoGain + ";Gain Ratio;" + gainRatio + ";Delta Gini Index;" + deltaGini + ";";
		System.out.println(line);
		
		return aux;
	}
	
	private static Double calculateEntropyFor2Bins(Bin bin1, Bin bin2) {
		BigDecimal entropyA = new BigDecimal(bin1.getEntropy());
		BigDecimal numInstancesA = new BigDecimal(bin1.getNumberOfInstances());
		BigDecimal weightedA = entropyA.multiply(numInstancesA);
		
		BigDecimal entropyB = new BigDecimal(bin2.getEntropy());
		BigDecimal numInstancesB = new BigDecimal(bin2.getNumberOfInstances());
		BigDecimal weightedB = entropyB.multiply(numInstancesB);
		
		BigDecimal weightedAB = weightedA.add(weightedB);
		BigDecimal totalInstances = numInstancesA.add(numInstancesB);
		
		BigDecimal weightedMean = weightedAB.divide(totalInstances, weightedAB.scale(), RoundingMode.DOWN);
		
		return weightedMean.doubleValue();
	}
	
	private static Double calculateEntropyFor3Bins(Bin bin1, Bin bin2, Bin bin3) {
		BigDecimal entropyA = new BigDecimal(bin1.getEntropy());
		BigDecimal numInstancesA = new BigDecimal(bin1.getNumberOfInstances());
		BigDecimal weightedA = entropyA.multiply(numInstancesA);
		
		BigDecimal entropyB = new BigDecimal(bin2.getEntropy());
		BigDecimal numInstancesB = new BigDecimal(bin2.getNumberOfInstances());
		BigDecimal weightedB = entropyB.multiply(numInstancesB);
		
		BigDecimal entropyC = new BigDecimal(bin3.getEntropy());
		BigDecimal numInstancesC = new BigDecimal(bin3.getNumberOfInstances());
		BigDecimal weightedC = entropyC.multiply(numInstancesC);
		
		BigDecimal weightedAB = weightedA.add(weightedB);
		BigDecimal weightedABC = weightedAB.add(weightedC);
		
		BigDecimal numInstancesAB = numInstancesA.add(numInstancesB);
		BigDecimal numInstancesABC = numInstancesAB.add(numInstancesC);
		
		BigDecimal weightedMean = weightedABC.divide(numInstancesABC, weightedABC.scale(), RoundingMode.DOWN);
		
		return weightedMean.doubleValue();
	}
	
	private static Double calculateEntropyFor4Bins(Bin bin1, Bin bin2, Bin bin3, Bin bin4) {
		BigDecimal entropyA = new BigDecimal(bin1.getEntropy());
		BigDecimal numInstancesA = new BigDecimal(bin1.getNumberOfInstances());
		BigDecimal weightedA = entropyA.multiply(numInstancesA);
		
		BigDecimal entropyB = new BigDecimal(bin2.getEntropy());
		BigDecimal numInstancesB = new BigDecimal(bin2.getNumberOfInstances());
		BigDecimal weightedB = entropyB.multiply(numInstancesB);
		
		BigDecimal entropyC = new BigDecimal(bin3.getEntropy());
		BigDecimal numInstancesC = new BigDecimal(bin3.getNumberOfInstances());
		BigDecimal weightedC = entropyC.multiply(numInstancesC);
		
		BigDecimal entropyD = new BigDecimal(bin4.getEntropy());
		BigDecimal numInstancesD = new BigDecimal(bin4.getNumberOfInstances());
		BigDecimal weightedD = entropyD.multiply(numInstancesD);
		
		BigDecimal weightedAB = weightedA.add(weightedB);
		BigDecimal weightedABC = weightedAB.add(weightedC);
		BigDecimal weightedABCD = weightedABC.add(weightedD);
		
		BigDecimal numInstancesAB = numInstancesA.add(numInstancesB);
		BigDecimal numInstancesABC = numInstancesAB.add(numInstancesC);
		BigDecimal numInstancesABCD = numInstancesABC.add(numInstancesD);
		
		BigDecimal weightedMean = weightedABCD.divide(numInstancesABCD, weightedABCD.scale(), RoundingMode.DOWN);
		
		return weightedMean.doubleValue();
	}
	
	private static Double calculateEntropyFor5Bins(Bin bin1, Bin bin2, Bin bin3, Bin bin4, Bin bin5) {
		BigDecimal entropyA = new BigDecimal(bin1.getEntropy());
		BigDecimal numInstancesA = new BigDecimal(bin1.getNumberOfInstances());
		BigDecimal weightedA = entropyA.multiply(numInstancesA);
		
		BigDecimal entropyB = new BigDecimal(bin2.getEntropy());
		BigDecimal numInstancesB = new BigDecimal(bin2.getNumberOfInstances());
		BigDecimal weightedB = entropyB.multiply(numInstancesB);
		
		BigDecimal entropyC = new BigDecimal(bin3.getEntropy());
		BigDecimal numInstancesC = new BigDecimal(bin3.getNumberOfInstances());
		BigDecimal weightedC = entropyC.multiply(numInstancesC);
		
		BigDecimal entropyD = new BigDecimal(bin4.getEntropy());
		BigDecimal numInstancesD = new BigDecimal(bin4.getNumberOfInstances());
		BigDecimal weightedD = entropyD.multiply(numInstancesD);
		
		BigDecimal entropyE = new BigDecimal(bin5.getEntropy());
		BigDecimal numInstancesE = new BigDecimal(bin5.getNumberOfInstances());
		BigDecimal weightedE = entropyE.multiply(numInstancesE);
		
		BigDecimal weightedAB = weightedA.add(weightedB);
		BigDecimal weightedABC = weightedAB.add(weightedC);
		BigDecimal weightedABCD = weightedABC.add(weightedD);
		BigDecimal weightedABCDE = weightedABCD.add(weightedE);
		
		BigDecimal numInstancesAB = numInstancesA.add(numInstancesB);
		BigDecimal numInstancesABC = numInstancesAB.add(numInstancesC);
		BigDecimal numInstancesABCD = numInstancesABC.add(numInstancesD);
		BigDecimal numInstancesABCDE = numInstancesABCD.add(numInstancesE);
		
		BigDecimal weightedMean = weightedABCDE.divide(numInstancesABCDE, weightedABCDE.scale(), RoundingMode.DOWN);
		
		return weightedMean.doubleValue();
	}
	
	private static double calculateInfoGainFor2Bins(Bin bin1, Bin bin2) {
		Bin joint = groupBins(bin1, bin2);
		double totalInstancesJoint = (double) joint.getNumberOfInstances();
		double[] totalInstancesPerClassJoint = joint.getQuantityOfInstancesPerClass();
		double infoD = 0d;
		for(int i = 0; i < totalInstancesPerClassJoint.length; i++)
		{
			double quantity = totalInstancesPerClassJoint[i];
			infoD += Metric.calculateInfoD(quantity, totalInstancesJoint);
		}
		
		double[] numInstancesPerClassBin1 = bin1.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin2 = bin2.getQuantityOfInstancesPerClass();
		
		double infoA = 0d;
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin1, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin2, totalInstancesJoint);
		
		double gain = infoD - infoA;
		
		return gain;
	}
	
	private static double calculateInfoGainFor3Bins(Bin bin1, Bin bin2, Bin bin3) {
		Bin joint12 = groupBins(bin1, bin2);
		Bin joint = groupBins(joint12, bin3);
		double totalInstancesJoint = (double) joint.getNumberOfInstances();
		double[] totalInstancesPerClassJoint = joint.getQuantityOfInstancesPerClass();
		double infoD = 0d;
		for(int i = 0; i < totalInstancesPerClassJoint.length; i++)
		{
			double quantity = totalInstancesPerClassJoint[i];
			infoD += Metric.calculateInfoD(quantity, totalInstancesJoint);
		}
		
		double[] numInstancesPerClassBin1 = bin1.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin2 = bin2.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin3 = bin3.getQuantityOfInstancesPerClass();
		
		double infoA = 0d;
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin1, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin2, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin3, totalInstancesJoint);
		
		double gain = infoD - infoA;
		
		return gain;
	}
	
	private static double calculateInfoGainFor4Bins(Bin bin1, Bin bin2, Bin bin3, Bin bin4) {
		Bin joint12 = groupBins(bin1, bin2);
		Bin joint123 = groupBins(joint12, bin3);
		Bin joint = groupBins(joint123, bin4);
		double totalInstancesJoint = (double) joint.getNumberOfInstances();
		double[] totalInstancesPerClassJoint = joint.getQuantityOfInstancesPerClass();
		double infoD = 0d;
		for(int i = 0; i < totalInstancesPerClassJoint.length; i++)
		{
			double quantity = totalInstancesPerClassJoint[i];
			infoD += Metric.calculateInfoD(quantity, totalInstancesJoint);
		}
		
		double[] numInstancesPerClassBin1 = bin1.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin2 = bin2.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin3 = bin3.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin4 = bin4.getQuantityOfInstancesPerClass();
		
		double infoA = 0d;
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin1, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin2, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin3, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin4, totalInstancesJoint);
		
		double gain = infoD - infoA;
		
		return gain;
	}
	
	private static double calculateInfoGainFor5Bins(Bin bin1, Bin bin2, Bin bin3, Bin bin4, Bin bin5) {
		Bin joint12 = groupBins(bin1, bin2);
		Bin joint123 = groupBins(joint12, bin3);
		Bin joint1234 = groupBins(joint123, bin4);
		Bin joint = groupBins(joint1234, bin5);
		double totalInstancesJoint = (double) joint.getNumberOfInstances();
		double[] totalInstancesPerClassJoint = joint.getQuantityOfInstancesPerClass();
		double infoD = 0d;
		for(int i = 0; i < totalInstancesPerClassJoint.length; i++)
		{
			double quantity = totalInstancesPerClassJoint[i];
			infoD += Metric.calculateInfoD(quantity, totalInstancesJoint);
		}
		
		double[] numInstancesPerClassBin1 = bin1.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin2 = bin2.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin3 = bin3.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin4 = bin4.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin5 = bin5.getQuantityOfInstancesPerClass();
		
		double infoA = 0d;
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin1, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin2, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin3, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin4, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin5, totalInstancesJoint);
		
		double gain = infoD - infoA;
		
		return gain;
	}
	
	private static double calculateGainRatioFor2Bins(Bin bin1, Bin bin2) {
		Bin joint = groupBins(bin1, bin2);
		double totalInstancesJoint = (double) joint.getNumberOfInstances();
		double[] totalInstancesPerClassJoint = joint.getQuantityOfInstancesPerClass();
		double infoD = 0d;
		for(int i = 0; i < totalInstancesPerClassJoint.length; i++)
		{
			double quantity = totalInstancesPerClassJoint[i];
			infoD += Metric.calculateInfoD(quantity, totalInstancesJoint);
		}
		
		double[] numInstancesPerClassBin1 = bin1.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin2 = bin2.getQuantityOfInstancesPerClass();
		
		double infoA = 0d;
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin1, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin2, totalInstancesJoint);
		
		double gain = infoD - infoA;
		
		double splitInfo = 0d;
		
		double numInstancesBin1 = (double) bin1.getNumberOfInstances();
		double numInstancesBin2 = (double) bin2.getNumberOfInstances();
		
		splitInfo += Metric.calculateSplitInfo(numInstancesBin1, totalInstancesJoint);
		splitInfo += Metric.calculateSplitInfo(numInstancesBin2, totalInstancesJoint);
		
		double gainRatio = Metric.calculateGainRatio(gain, splitInfo);
		
		return gainRatio;
	}
	
	private static double calculateGainRatioFor3Bins(Bin bin1, Bin bin2, Bin bin3) {
		Bin joint12 = groupBins(bin1, bin2);
		Bin joint = groupBins(joint12, bin3);
		double totalInstancesJoint = (double) joint.getNumberOfInstances();
		double[] totalInstancesPerClassJoint = joint.getQuantityOfInstancesPerClass();
		double infoD = 0d;
		for(int i = 0; i < totalInstancesPerClassJoint.length; i++)
		{
			double quantity = totalInstancesPerClassJoint[i];
			infoD += Metric.calculateInfoD(quantity, totalInstancesJoint);
		}
		
		double[] numInstancesPerClassBin1 = bin1.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin2 = bin2.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin3 = bin3.getQuantityOfInstancesPerClass();
		
		double infoA = 0d;
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin1, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin2, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin3, totalInstancesJoint);
		
		double gain = infoD - infoA;
		
		double splitInfo = 0d;
		
		double numInstancesBin1 = (double) bin1.getNumberOfInstances();
		double numInstancesBin2 = (double) bin2.getNumberOfInstances();
		double numInstancesBin3 = (double) bin3.getNumberOfInstances();
		
		splitInfo += Metric.calculateSplitInfo(numInstancesBin1, totalInstancesJoint);
		splitInfo += Metric.calculateSplitInfo(numInstancesBin2, totalInstancesJoint);
		splitInfo += Metric.calculateSplitInfo(numInstancesBin3, totalInstancesJoint);
		
		double gainRatio = Metric.calculateGainRatio(gain, splitInfo);
		
		return gainRatio;
	}
	
	private static double calculateGainRatioFor4Bins(Bin bin1, Bin bin2, Bin bin3, Bin bin4) {
		Bin joint12 = groupBins(bin1, bin2);
		Bin joint123 = groupBins(joint12, bin3);
		Bin joint = groupBins(joint123, bin4);
		double totalInstancesJoint = (double) joint.getNumberOfInstances();
		double[] totalInstancesPerClassJoint = joint.getQuantityOfInstancesPerClass();
		double infoD = 0d;
		for(int i = 0; i < totalInstancesPerClassJoint.length; i++)
		{
			double quantity = totalInstancesPerClassJoint[i];
			infoD += Metric.calculateInfoD(quantity, totalInstancesJoint);
		}
		
		double[] numInstancesPerClassBin1 = bin1.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin2 = bin2.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin3 = bin3.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin4 = bin4.getQuantityOfInstancesPerClass();
		
		double infoA = 0d;
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin1, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin2, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin3, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin4, totalInstancesJoint);
		
		double gain = infoD - infoA;
		
		double splitInfo = 0d;
		
		double numInstancesBin1 = (double) bin1.getNumberOfInstances();
		double numInstancesBin2 = (double) bin2.getNumberOfInstances();
		double numInstancesBin3 = (double) bin3.getNumberOfInstances();
		double numInstancesBin4 = (double) bin4.getNumberOfInstances();
		
		splitInfo += Metric.calculateSplitInfo(numInstancesBin1, totalInstancesJoint);
		splitInfo += Metric.calculateSplitInfo(numInstancesBin2, totalInstancesJoint);
		splitInfo += Metric.calculateSplitInfo(numInstancesBin3, totalInstancesJoint);
		splitInfo += Metric.calculateSplitInfo(numInstancesBin4, totalInstancesJoint);
		
		double gainRatio = Metric.calculateGainRatio(gain, splitInfo);
		
		return gainRatio;
	}
	
	private static double calculateGainRatioFor5Bins(Bin bin1, Bin bin2, Bin bin3, Bin bin4, Bin bin5) {
		Bin joint12 = groupBins(bin1, bin2);
		Bin joint123 = groupBins(joint12, bin3);
		Bin joint1234 = groupBins(joint123, bin4);
		Bin joint = groupBins(joint1234, bin5);
		double totalInstancesJoint = (double) joint.getNumberOfInstances();
		double[] totalInstancesPerClassJoint = joint.getQuantityOfInstancesPerClass();
		double infoD = 0d;
		for(int i = 0; i < totalInstancesPerClassJoint.length; i++)
		{
			double quantity = totalInstancesPerClassJoint[i];
			infoD += Metric.calculateInfoD(quantity, totalInstancesJoint);
		}
		
		double[] numInstancesPerClassBin1 = bin1.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin2 = bin2.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin3 = bin3.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin4 = bin4.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin5 = bin5.getQuantityOfInstancesPerClass();
		
		double infoA = 0d;
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin1, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin2, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin3, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin4, totalInstancesJoint);
		infoA += Metric.calculateInfoAttribute(numInstancesPerClassBin5, totalInstancesJoint);
		
		double gain = infoD - infoA;
		
		double splitInfo = 0d;
		
		double numInstancesBin1 = (double) bin1.getNumberOfInstances();
		double numInstancesBin2 = (double) bin2.getNumberOfInstances();
		double numInstancesBin3 = (double) bin3.getNumberOfInstances();
		double numInstancesBin4 = (double) bin4.getNumberOfInstances();
		double numInstancesBin5 = (double) bin5.getNumberOfInstances();
		
		splitInfo += Metric.calculateSplitInfo(numInstancesBin1, totalInstancesJoint);
		splitInfo += Metric.calculateSplitInfo(numInstancesBin2, totalInstancesJoint);
		splitInfo += Metric.calculateSplitInfo(numInstancesBin3, totalInstancesJoint);
		splitInfo += Metric.calculateSplitInfo(numInstancesBin4, totalInstancesJoint);
		splitInfo += Metric.calculateSplitInfo(numInstancesBin5, totalInstancesJoint);
		
		double gainRatio = Metric.calculateGainRatio(gain, splitInfo);
		
		return gainRatio;
	}
	
	private static double calculateDeltaGiniIndexFor2Bins(Bin bin1, Bin bin2) {
		Bin joint = groupBins(bin1, bin2);
		double totalInstancesJoint = (double) joint.getNumberOfInstances();
		double[] totalInstancesPerClassJoint = joint.getQuantityOfInstancesPerClass();
		
		double giniD = Metric.calculateGiniD(totalInstancesPerClassJoint);
		
		double giniA = 0d;
		
		double[] numInstancesPerClassBin1 = bin1.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin2 = bin2.getQuantityOfInstancesPerClass();
		
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin1, totalInstancesJoint);
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin2, totalInstancesJoint);
		
		double deltaGiniIndex = Metric.calculateDeltaGini(giniD, giniA);
		
		return deltaGiniIndex;
	}
	
	private static double calculateDeltaGiniIndexFor3Bins(Bin bin1, Bin bin2, Bin bin3) {
		Bin joint12 = groupBins(bin1, bin2);
		Bin joint = groupBins(joint12, bin3);
		double totalInstancesJoint = (double) joint.getNumberOfInstances();
		double[] totalInstancesPerClassJoint = joint.getQuantityOfInstancesPerClass();
		
		double giniD = Metric.calculateGiniD(totalInstancesPerClassJoint);
		
		double giniA = 0d;
		
		double[] numInstancesPerClassBin1 = bin1.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin2 = bin2.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin3 = bin3.getQuantityOfInstancesPerClass();
		
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin1, totalInstancesJoint);
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin2, totalInstancesJoint);
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin3, totalInstancesJoint);
		
		double deltaGiniIndex = Metric.calculateDeltaGini(giniD, giniA);
		
		return deltaGiniIndex;
	}
	
	private static double calculateDeltaGiniIndexFor4Bins(Bin bin1, Bin bin2, Bin bin3, Bin bin4) {
		Bin joint12 = groupBins(bin1, bin2);
		Bin joint123 = groupBins(joint12, bin3);
		Bin joint = groupBins(joint123, bin4);
		double totalInstancesJoint = (double) joint.getNumberOfInstances();
		double[] totalInstancesPerClassJoint = joint.getQuantityOfInstancesPerClass();
		
		double giniD = Metric.calculateGiniD(totalInstancesPerClassJoint);
		
		double giniA = 0d;
		
		double[] numInstancesPerClassBin1 = bin1.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin2 = bin2.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin3 = bin3.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin4 = bin4.getQuantityOfInstancesPerClass();
		
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin1, totalInstancesJoint);
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin2, totalInstancesJoint);
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin3, totalInstancesJoint);
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin4, totalInstancesJoint);
		
		double deltaGiniIndex = Metric.calculateDeltaGini(giniD, giniA);
		
		return deltaGiniIndex;
	}
	
	private static double calculateDeltaGiniIndexFor5Bins(Bin bin1, Bin bin2, Bin bin3, Bin bin4, Bin bin5) {
		Bin joint12 = groupBins(bin1, bin2);
		Bin joint123 = groupBins(joint12, bin3);
		Bin joint1234 = groupBins(joint123, bin4);
		Bin joint = groupBins(joint1234, bin5);
		double totalInstancesJoint = (double) joint.getNumberOfInstances();
		double[] totalInstancesPerClassJoint = joint.getQuantityOfInstancesPerClass();
		
		double giniD = Metric.calculateGiniD(totalInstancesPerClassJoint);
		
		double giniA = 0d;
		
		double[] numInstancesPerClassBin1 = bin1.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin2 = bin2.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin3 = bin3.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin4 = bin4.getQuantityOfInstancesPerClass();
		double[] numInstancesPerClassBin5 = bin5.getQuantityOfInstancesPerClass();
		
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin1, totalInstancesJoint);
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin2, totalInstancesJoint);
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin3, totalInstancesJoint);
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin4, totalInstancesJoint);
		giniA += Metric.calculateGiniAttribute(numInstancesPerClassBin5, totalInstancesJoint);
		
		double deltaGiniIndex = Metric.calculateDeltaGini(giniD, giniA);
		
		return deltaGiniIndex;
	}
	
	private static Double getMinimumAttributeValue(Instances dataset, int attributeIndex) {
		if(dataset.attribute(attributeIndex).isNumeric())
		{
			double minimum = Double.POSITIVE_INFINITY;
			for(Instance instance : dataset)
			{
				if(!Double.isNaN(instance.value(attributeIndex)))
				{
					if(Double.compare(minimum, instance.value(attributeIndex)) > 0)
					{
						minimum = instance.value(attributeIndex);
					}
				}
			}
			
			// checks if minimum received at least one value
			if(Double.compare(minimum, Double.POSITIVE_INFINITY) == 0)
			{
				return Double.NaN;
			}
			
			return minimum;
		}
		else
		{
			return Double.NaN;
		}
	}
	
	private static Double getMaximumAttributeValue(Instances dataset, int attributeIndex) {
		if(dataset.attribute(attributeIndex).isNumeric())
		{
			double maximum = Double.NEGATIVE_INFINITY;
			for(Instance instance : dataset)
			{
				if(!Double.isNaN(instance.value(attributeIndex)))
				{
					if(Double.compare(maximum, instance.value(attributeIndex)) < 0)
					{
						maximum = instance.value(attributeIndex);
					}
				}
			}
			
			// checks if maximum received at least one value
			if(Double.compare(maximum, Double.NEGATIVE_INFINITY) == 0)
			{
				return Double.NaN;
			}
			
			return maximum;
		}
		else
		{
			return Double.NaN;
		}
	}
	
	private static Map<Integer, Double[]> getRanges(BigDecimal minimum, BigDecimal maximum, BigDecimal rangeSize, int total) {
		// the number of the range and its values
		Map<Integer, Double[]> ranges = new HashMap<>();
		
		BigDecimal begin = minimum, end = minimum;
		
		for(int range = 0; range < total; range++)
		{
			begin = end;
			end = end.add(rangeSize);
			
			if(range == total - 1)
			{
				end = maximum;
			}
			
			Double values[] = {begin.doubleValue(), end.doubleValue()};
			ranges.put(range, values);
		}
		
		return ranges;
	}
	
	private static List<Bin> getHistogram(Map<Integer, Double[]> ranges, Instances dataset, int attributeIndex) {
		List<Bin> bins = new ArrayList<>();
		
		for(int index = 0; index < dataset.numInstances(); index++)
		{
			List<Instance> instances = new ArrayList<>();
			
			Double range[] = ranges.get(index);
			double begin = range[0];
			double end = range[1];
			
			for(int instance = 0; instance < dataset.numInstances(); instance++)
			{
				double attributeValue = dataset.get(instance).value(attributeIndex);
				
				// checks if it is the last range
				if(index == dataset.numInstances() - 1)
				{
					if(Utility.isGreaterThanOrEqualTo(attributeValue, begin))
					{
						instances.add(dataset.get(instance));
					}
				}
				else
				{
					if(Utility.isGreaterThanOrEqualTo(attributeValue, begin) && !Utility.isGreaterThanOrEqualTo(attributeValue, end))
					{
						instances.add(dataset.get(instance));
					}
				}
				
			}
			
			Bin bin = new Bin(instances, attributeIndex, begin, end);
			bins.add(bin);
		}
		
		return bins;
	}
	
	private static Instances buildDataset(List<Bin> bins, Instances dataset) {
		Bin bin = bins.get(0);
		int attributeIndex = bin.getAttributeIndex(); 
		
		List<String> attributeValues = getDiscretizationRanges(bins);
		ArrayList<Attribute> attributes = new ArrayList<>();
		
		for(int i = 0; i < dataset.numAttributes(); i++)
		{
			Attribute attribute = dataset.attribute(i);
			
			if(i == attributeIndex)
			{
				Attribute discretizedAttribute = new Attribute(attribute.name(), attributeValues);
				attributes.add(discretizedAttribute);
			}
			else
			{
				attributes.add(attribute);
			}
			
		}
		
		Instances discretized = new Instances(dataset.relationName(), attributes, dataset.numInstances());
		
		for(int b = 0; b < bins.size(); b++)
		{
			bin = bins.get(b);
			List<Instance> instances = bin.getInstances();
			for(Instance instance : instances)
			{
				String attributeValue = attributeValues.get(b);
				instance.setDataset(discretized);
				instance.setValue(attributeIndex, attributeValue);
				discretized.add(instance);
			}
		}
		
		discretized.setClassIndex(discretized.numAttributes() - 1);
		
		return discretized;
	}
	
	private static Integer getNumberOfBinInstances(List<Bin> bins) {
		int count = 0;
		for(int index = 0; index < bins.size(); index++)
		{
			Bin bin = bins.get(index);
			count += bin.getNumberOfInstances();
		}
		
		return count;
	}
	
	private static List<Bin> removeEmptyBins(List<Bin> bins, double rangeSize) {
		List<Bin> fullBins = new ArrayList<>();
		
		for(int index = 0; index < bins.size(); index++)
		{
			Bin bin = bins.get(index);
			if(bin.getNumberOfInstances() > 0)
			{
				fullBins.add(bin);
			}
		}
		
		adjustRanges(fullBins, rangeSize);
		
		return fullBins;
	}
	
	private static void adjustRanges(List<Bin> bins, double rangeSize) {
		for(int index = 0; index < bins.size() - 1; index++)
		{
			Bin x = bins.get(index);
			Bin y = bins.get(index + 1);
			
			Double xEnd = x.getEnd();
			Double yStart = y.getStart();
			
			// checks if there were empty bins between them
			if(Double.compare(Utility.truncate((yStart - xEnd)), rangeSize) < 0)
			{
				x.setEnd(yStart);
				y.setStart(yStart);
			}
			else
			{
				Double averageXY = (double) (xEnd + yStart) / 2;
				averageXY = Utility.roundDouble(averageXY);
				x.setEnd(averageXY);
				y.setStart(averageXY);
			}
		}
	}
	
	private static Bin groupBins(Bin x, Bin y) {
		List<Instance> instances = new ArrayList<>();
		List<Instance> xInst = x.getInstances();
		List<Instance> yInst = y.getInstances();
		
		for(int index = 0; index <xInst.size(); index++)
		{
			Instance instance = xInst.get(index);
			instances.add(instance);
		}
		for(int index = 0; index < yInst.size(); index++)
		{
			Instance instance = yInst.get(index);
			instances.add(instance);
		}
		
		Bin xy = new Bin(instances, x.getAttributeIndex(), x.getStart(), y.getEnd());
		
		return xy;
	}
	
	private static List<String> getDiscretizationRanges(List<Bin> bins) {
		List<String> ranges = new ArrayList<>();
		
		double start = 0, end = 0;
		
		for(int index = 0; index < bins.size(); index++)
		{
			Bin bin = bins.get(index);
			
			if(index == 0)
			{
				start = bin.getStart();
				end = bin.getEnd();
			}
			else
			{
				start = end;
				end = bin.getEnd();
			}
			
			String range = start + "/" + end;
			ranges.add(range);
		}
		
		return ranges;
	}

}
