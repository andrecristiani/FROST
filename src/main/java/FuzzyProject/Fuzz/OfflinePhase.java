package FuzzyProject.Fuzz;

import FuzzyProject.Fuzz.Models.Ensemble;
import FuzzyProject.Fuzz.Models.Evaluation.ClassicMeasures;
import weka.core.Instances;

public class OfflinePhase {
    public Ensemble inicializar(String dataset, String caminho, int tComite, Instances trainSet, double fuzzification, double alpha, double theta, int C, int K, int minWeight) throws Exception {
        Ensemble ensemble = new Ensemble(dataset, caminho, tComite, fuzzification, alpha, theta, C, K, minWeight);
        long start = System.currentTimeMillis();
        ensemble.trainInitialEnsemble(trainSet);
        long elapsed = System.currentTimeMillis() - start;
        double timeInSeconds = Double.parseDouble(String.valueOf(elapsed))/1000;
        ensemble.timeOffline = timeInSeconds;
        return ensemble;
    }
}
