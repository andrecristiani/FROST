package FuzzyProject;

import FuzzyProject.Fuzz.Models.Ensemble;
import FuzzyProject.Fuzz.Models.Evaluation.ClassicMeasures;
import FuzzyProject.Fuzz.Models.MaxTipicity;
import FuzzyProject.Fuzz.OfflinePhase;
import FuzzyProject.Fuzz.OnlinePhase;
import FuzzyProject.Fuzz.Utils.HandlesFiles;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FuzzySystem {
    public static void main(String[] args) throws IOException, Exception {
        String dataset = "unsw";
        String caminho = "";
        String current = (new File(".")).getCanonicalPath();
        caminho = current + "/" + dataset + "/";
        double fuzzyfication = 2;
        double alpha = 2;
        double theta = 1;

        ConverterUtils.DataSource source1;
        Instances data1;

        source1 = new ConverterUtils.DataSource(caminho + dataset + "-train.arff");
        data1 = source1.getDataSet();
        data1.setClassIndex(data1.numAttributes() - 1);

        List<Instances> chunks = new ArrayList<>();
        chunks.add(data1);

        List<ClassicMeasures> classicMeasuresList = new ArrayList<>();

        for(int i=0; i<33; i++) {
            OfflinePhase offlinePhase = new OfflinePhase();
            Ensemble ensemble = offlinePhase.inicializar(dataset, current + "/" + dataset + "/", 12, data1, fuzzyfication, alpha, theta, /*141657*/112148, 16, 10);
            OnlinePhase onlinePhase = new OnlinePhase();
            ensemble.allTipMax = new MaxTipicity(0.70);
            ensemble.thetaAdapter = 0.60;
            ensemble.N = 2;
            ClassicMeasures c = onlinePhase.initialize(current + "/" + dataset + "/", dataset, ensemble, 2000, 15000, 40, 4, 0.5);
            classicMeasuresList.add(c);
        }
        HandlesFiles.salvaPredicoes(classicMeasuresList, dataset);
    }
}



