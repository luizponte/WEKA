package main.permutation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Map.Entry;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class Permutation {
	
static long seed;
	
	final static String permuted_dataset = "PERMUTED_DATASET";
	final static String original_dataset = "ORIGINAL_DATASET";
	
	final static String metric_pctCorrect = "pctCorrect";
	final static String metric_pctIncorrect = "pctIncorrect";
	final static String metric_correct = "correct";
	final static String metric_incorrect = "incorrect";
	final static String metric_fMeasureA = "fMeasureA";
	final static String metric_fMeasureB = "fMeasureB";
	final static String metric_weightedFMeasure = "weightedFMeasure";
	final static String metric_meanAbsoluteError = "meanAbsoluteError";
	final static String metric_rootMeanSquaredError = "rootMeanSquaredError";
	final static String metric_relativeAbsoluteError = "relativeAbsoluteError";
	final static String metric_rootRelativeSquaredError = "rootRelativeSquaredError";
	final static String metric_weightedPrecision = "weightedPrecision";
	final static String metric_weightedRecall = "weightedRecall";
	final static String metric_weightedAUCROC = "weightedAUCROC";
	final static String metric_correlationCoefficient = "correlationCoefficient";
	
	static Random random;
	
	public static Map<String, Double> permutationTest(Instances train, Instances test, int permutations, int crossValidationFolds, int attributeIndex, AbstractClassifier classifier, long seed) {
		setSeed(seed);
		List<Instances> permutedDatasets = getPermutedDatasets(train, attributeIndex, permutations, random);
		Map<String, List<Evaluation>> evaluationModels = null;
		try {
			evaluationModels = buildEvaluationModels(train, test, permutedDatasets, classifier, permutations, crossValidationFolds);
		} catch (Exception e) {
			// the class was not defined
			e.printStackTrace();
		}
		boolean isRegression = train.classAttribute().isNumeric();
		Map<String, Double> pValues = null;
		try {
			pValues = calculatePValues(evaluationModels, isRegression, permutations);
		} catch (Exception e) {
			// the evaluation model of original dataset was not found
			e.printStackTrace();
		}
		return pValues;
	}
	
	private static void setSeed(long seedValue) {
		seed = seedValue;
		random = new Random(seed);
	}
	
	private static List<Integer> shuffleIndexes(Random random, int size) {
		List<Integer> values = new ArrayList<>();
		
		while(values.size() < size)
		{
			// generates values between 0 and the specified maximal value
			Integer value = random.nextInt(size);
			if(!values.contains(value))
			{
				// avoids duplicates
				values.add(value);
			}
		}
		
		return values;
	}
	
	private static List<Instances> getPermutedDatasets(Instances original, int attributeIndex, int permutations, Random random) {
		List<Instances> permuted = new ArrayList<>();
		
		for(int permutation = 1; permutation <= permutations; permutation++)
		{
			Instances permutedDataset = permuteAttributeValues(original, attributeIndex, random);
			permuted.add(permutedDataset);
		}
		
		return permuted;
	}
	
	private static Instances permuteAttributeValues(Instances original, int attributeIndex, Random random) {
		Instances permuted = new Instances(original);
		
		List<Integer> shuffledIndexes = new ArrayList<>();
		if(original.numInstances() == 1)
		{
			// will not permute because there is only one instance
			shuffledIndexes.add(0);
		}
		else
		{
			shuffledIndexes = shuffleIndexes(random, original.numInstances());
		}
		
		for(int index = 0; index < original.numInstances(); index++)
		{
			Instance instanceP = permuted.instance(index);
			
			Instance instanceO = original.instance(shuffledIndexes.get(index));
			double attributeO = instanceO.value(attributeIndex);
			
			instanceP.setValue(attributeIndex, attributeO);
			
			permuted.set(index, instanceP);
		}
		
		return permuted;
	}
	
	private static Map<String, List<Evaluation>> buildEvaluationModels(Instances original, Instances test, List<Instances> permutedDatasets, AbstractClassifier classifier, int permutations, int folds) throws Exception {
		Evaluation evalOriginal;
		if(folds > 0)
		{
			evalOriginal = createEvaluatorWithCV(classifier, original, folds); 
		}
		else
		{
			evalOriginal = createEvaluatorWithPartition(classifier, original, test);
		}
		
		List<Evaluation> modelEvaluationOriginal = new ArrayList<>();
		modelEvaluationOriginal.add(evalOriginal);
		
		Evaluation evalPermuted = null;
		List<Evaluation> modelEvaluationPermuted = new ArrayList<>();
		
		Classifier classifierForPermuted = AbstractClassifier.makeCopy(classifier);
		
		Instances permuted = null;
		
		for(int index = 0; index < permutations; index++)
		{
			permuted = permutedDatasets.get(index);
			classifierForPermuted.buildClassifier(permuted);
			
			if(folds > 0)
			{
				evalPermuted = createEvaluatorWithCV(classifierForPermuted, permuted, folds);
			}
			else
			{
				evalPermuted = createEvaluatorWithPartition(classifierForPermuted, permuted, test);
			}
			
			modelEvaluationPermuted.add(evalPermuted);
			
		}
		
		Map<String, List<Evaluation>> evaluationModels = new HashMap<>();
		evaluationModels.put(original_dataset, modelEvaluationOriginal);
		evaluationModels.put(permuted_dataset, modelEvaluationPermuted);
		
		return evaluationModels;
	}
	
	private static Map<String, Double> calculatePValues(Map<String, List<Evaluation>> evaluationModels, boolean isRegression, int permutations) throws Exception {
		
		List<Evaluation> evalOriginalList = null;
		List<Evaluation> evalPermutedList = null;
		
		Evaluation evalOriginal = null;
		
		if(evaluationModels.containsKey(original_dataset))
		{
			evalOriginalList = evaluationModels.get(original_dataset);
			evalOriginal = evalOriginalList.get(0);
		}
		else
		{
			throw new Exception("Evaluation for the original model was not found!");
		}
		
		evalPermutedList = evaluationModels.get(permuted_dataset);
		
		Map<String, Integer> metrics = new HashMap<>();
		initializeMetricsMap(metrics, isRegression);
		
		double correct = 0, incorrect = 0, numCorrect = 0, numIncorrect = 0, correlation = 0;
		double averageFMeasure = 0, averagePrecision = 0, averageRecall = 0, averageAUCROC = 0;
		
		if(!isRegression)
		{
			correct = evalOriginal.pctCorrect();
			incorrect = evalOriginal.pctIncorrect();
			
			numCorrect = evalOriginal.correct();
			numIncorrect = evalOriginal.incorrect();
			
			averageFMeasure = evalOriginal.weightedFMeasure();
			
			averagePrecision = evalOriginal.weightedPrecision();
			averageRecall = evalOriginal.weightedRecall();
			
			averageAUCROC = evalOriginal.weightedAreaUnderROC();
		}
		
		double meanAbsoluteError = evalOriginal.meanAbsoluteError();
		double rmse = evalOriginal.rootMeanSquaredError();
		double rae = evalOriginal.relativeAbsoluteError();
		double rrse = evalOriginal.rootRelativeSquaredError();
		
		if(isRegression)
		{
			correlation = evalOriginal.correlationCoefficient();
		}
		
		for(int index = 0; index < permutations; index++)
		{
			
			Evaluation evalPermuted = null;
			int metric = 0;
			
			evalPermuted = evalPermutedList.get(index);
				
			if(!isRegression && evalPermuted.pctCorrect() >= correct)
			{
				metric = metrics.get(metric_pctCorrect);
				metrics.put(metric_pctCorrect, metric + 1);
			}
				
			if(!isRegression && evalPermuted.pctIncorrect() <= incorrect)
			{
				metric = metrics.get(metric_pctIncorrect);
				metrics.put(metric_pctIncorrect, metric + 1);
			}
				
			if(!isRegression && evalPermuted.correct() >= numCorrect)
			{
				metric = metrics.get(metric_correct);
				metrics.put(metric_correct, metric + 1);
			}
				
			if(!isRegression && evalPermuted.incorrect() <= numIncorrect)
			{
				metric = metrics.get(metric_incorrect);
				metrics.put(metric_incorrect, metric + 1);
			}
				
			if(!isRegression && evalPermuted.weightedFMeasure() >= averageFMeasure)
			{
				metric = metrics.get(metric_weightedFMeasure);
				metrics.put(metric_weightedFMeasure, metric + 1);
			}
				
			if(!isRegression && evalPermuted.weightedPrecision() >= averagePrecision)
			{
				metric = metrics.get(metric_weightedPrecision);
				metrics.put(metric_weightedPrecision, metric + 1);
			}
				
			if(!isRegression && evalPermuted.weightedRecall() >= averageRecall)
			{
				metric = metrics.get(metric_weightedRecall);
				metrics.put(metric_weightedRecall, metric + 1);
			}
				
			if(!isRegression && evalPermuted.weightedAreaUnderROC() >= averageAUCROC)
			{
				metric = metrics.get(metric_weightedAUCROC);
				metrics.put(metric_weightedAUCROC, metric + 1);
			}
				
			if(isRegression && evalPermuted.correlationCoefficient() >= correlation)
			{
				metric = metrics.get(metric_correlationCoefficient);
				metrics.put(metric_correlationCoefficient, metric + 1);
			}
				
			if(evalPermuted.meanAbsoluteError() <= meanAbsoluteError)
			{
				metric = metrics.get(metric_meanAbsoluteError);
				metrics.put(metric_meanAbsoluteError, metric + 1);
			}
				
			if(evalPermuted.rootMeanSquaredError() <= rmse)
			{
				metric = metrics.get(metric_rootMeanSquaredError);
				metrics.put(metric_rootMeanSquaredError, metric + 1);
			}
				
			if(evalPermuted.relativeAbsoluteError() <= rae)
			{
				metric = metrics.get(metric_relativeAbsoluteError);
				metrics.put(metric_relativeAbsoluteError, metric + 1);
			}
				
			if(evalPermuted.rootRelativeSquaredError() <= rrse)
			{
				metric = metrics.get(metric_rootRelativeSquaredError);
				metrics.put(metric_rootRelativeSquaredError, metric + 1);
			}
			
		}
		
		Map<String, Double> metricPValues = new HashMap<>();
		
		Set<Entry<String, Integer>> entrySet = metrics.entrySet();
		
		for(Entry<String, Integer> entry : entrySet) {
			String metric = entry.getKey();
			double pValue = getPValue(entry.getValue(), permutations);
			metricPValues.put(metric, pValue);
		}
		
		return metricPValues;
	}
	
	private static void initializeMetricsMap(Map<String, Integer> metricsMap, boolean isRegression) {
		
		if(isRegression)
		{
			metricsMap.put(metric_correlationCoefficient, 0);
		}
		else
		{
			metricsMap.put(metric_pctCorrect, 0);
			metricsMap.put(metric_pctIncorrect, 0);
			metricsMap.put(metric_correct, 0);
			metricsMap.put(metric_incorrect, 0);
			metricsMap.put(metric_fMeasureA, 0);
			metricsMap.put(metric_fMeasureB, 0);
			metricsMap.put(metric_weightedFMeasure, 0);
			metricsMap.put(metric_weightedPrecision, 0);
			metricsMap.put(metric_weightedRecall, 0);
			metricsMap.put(metric_weightedAUCROC, 0);
		}
		
		metricsMap.put(metric_meanAbsoluteError, 0);
		metricsMap.put(metric_rootMeanSquaredError, 0);
		metricsMap.put(metric_relativeAbsoluteError, 0);
		metricsMap.put(metric_rootRelativeSquaredError, 0);
	}
	
	private static double getPValue(int numberOfInstances, int permutations) {
		if(numberOfInstances == 0)
		{
			// performs the correction for the minimum probability value
			numberOfInstances = 1;
		}
		return (double) numberOfInstances / permutations;
		
	}
	
	private static Evaluation createEvaluatorWithCV(Classifier classifier, Instances dataset, int folds) throws Exception {
		classifier.buildClassifier(dataset);
		Evaluation eval = new Evaluation(dataset); // throws an exception if the class was not defined
		eval.crossValidateModel(classifier, dataset, folds, new Random(1));
		return eval;
	}
	
	private static Evaluation createEvaluatorWithPartition(Classifier classifier, Instances train, Instances test) throws Exception {
		classifier.buildClassifier(train);
		Evaluation eval = new Evaluation(train); // throws an exception if the class was not defined
		eval.evaluateModel(classifier, test);
		return eval;
	}

}
