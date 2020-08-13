package main.classification;

import java.util.HashMap;
import java.util.Map;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class Classification {
	
	public static NaiveBayes createDefaultNaiveBayes(Instances instances) throws Exception {
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(instances);
		return nb;
	}
	
	public static J48 createDefaultJ48(Instances instances) throws Exception {
		J48 j48 = new J48();
		j48.buildClassifier(instances);
		return j48;
	}
	
	public static RandomForest createDefaultRandomForest(Instances instances) throws Exception {
		RandomForest rf = new RandomForest();
		rf.buildClassifier(instances);
		return rf;
	}
	
	public static SMO createDefaultSMO(Instances instances) throws Exception {
		SMO smo = new SMO();
		smo.buildClassifier(instances);
		return smo;
	}
	
	public static IBk createKNN(Instances instances, String[] options) throws Exception {
		IBk knn = new IBk();
		knn.setOptions(options);
		knn.buildClassifier(instances);
		return knn;
	}
	
	public static Map<String, AbstractClassifier> createClassifiers(Instances instances, int neighbors) throws Exception {
		Map<String, AbstractClassifier> classifiers = new HashMap<>();
		
		NaiveBayes nb = createDefaultNaiveBayes(instances);
		classifiers.put("Naive Bayes",nb);
		
		SMO smo = createDefaultSMO(instances);
		classifiers.put("SMO PolyKernel",smo);
		
		for(int k = 1; k <= neighbors; k++) {
			String[] options = new String[2];
			options[0] = "-K";
			options[1] = "" + k;
			
			IBk knn = createKNN(instances, options);
			if(k < 10)
			{
				classifiers.put("IBk k = 0" + k, knn);
			}
			else
			{
				classifiers.put("IBk k = " + k, knn);
			}
		}
		
		// if you want, insert here more classifiers
		
		return classifiers;
	}

}
