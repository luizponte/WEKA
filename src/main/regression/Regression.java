package main.regression;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class Regression {
	
	public static LinearRegression createLinearRegression(Instances instances) throws Exception {
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(instances);
		return lr;
	}
	
	public static SMOreg createSMOReg(Instances instances) throws Exception {
		SMOreg smoReg = new SMOreg();
		smoReg.buildClassifier(instances);
		return smoReg;
	}
	
	public static IBk createKNN(Instances instances, String[] options) throws Exception {
		IBk knn = new IBk();
		knn.setOptions(options);
		knn.buildClassifier(instances);
		return knn;
	}
	
	public static Evaluation createEvaluator(Classifier classifier, Instances train, Instances test) throws Exception {
		Evaluation eval = new Evaluation(train);
		eval.evaluateModel(classifier, test);
		return eval;
		
	}
	
	public static Evaluation createEvaluatorWithCV(Classifier classifier, Instances dataset, int folds) throws Exception {
		Evaluation eval = new Evaluation(dataset);
		eval.crossValidateModel(classifier, dataset, folds, new Random(1));
		return eval;
		
	}
	
	public static Map<String,AbstractClassifier> createClassifiers(Instances instances, int neighbors) throws Exception {
		Map<String,AbstractClassifier> classifiers = new HashMap<>();
		
		// you will need a jar file to use the linear regression
		LinearRegression lr = createLinearRegression(instances);
		classifiers.put("LinearRegression", lr);
		
		SMOreg smoReg = createSMOReg(instances);
		classifiers.put("SMOreg", smoReg);
		
		for(int k = 1; k <= neighbors; k++) {
			String[] options = new String[2];
			options[0] = "-K";
			options[1] = "" + k;
			
			IBk knn = createKNN(instances, options);
			classifiers.put("IBk k = " + k, knn);
		}
		
		return classifiers;
	}

}
