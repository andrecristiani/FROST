package FuzzyProject.Fuzz;

import FuzzyProject.Fuzz.Models.Ensemble;
import FuzzyProject.Fuzz.Models.*;
import FuzzyProject.Fuzz.Models.Evaluation.ClassicMeasures;
import FuzzyProject.Fuzz.Utils.DistanceMeasures;
import FuzzyProject.Fuzz.Utils.FuzzyFunctions;
import FuzzyProject.Fuzz.Utils.HandlesFiles;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.FuzzyKMeansClusterer;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.IOException;
import java.util.*;

public class OnlinePhase {

    int acertos = 0;
    int acertosTotal = 0;
    int erros = 0;
    int errosTotal = 0;
    Ensemble ensemble;
    public Map<Double, Integer> unkRi = new HashMap<>();
    int novidadesClassificadas = 0;
    int classifiedExamples = 0;
    public Map<Double, Integer> exc = new HashMap<>();
    public int fp = 0;
    public int fn = 0;
    public int fe = 0;
    public int nc = 0;
    public int n = 0;

    public int ncTest = 0;
    public int ncTotal = 0;

    public void initialize(String caminho, String dataset, Ensemble comite, int latencia, int tChunk, int T, int kShort, double phi, int nExecution) throws IOException {
        ArrayList<ClassicMeasures> classicMeasuresArrayList = new ArrayList<>();
        this.ensemble = comite;
        DataSource source;
        Instances data;
        long start = 0;
        try {
            source = new DataSource(caminho + dataset + "-instances.arff");
            data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            ArrayList<Attribute> atts = new ArrayList<>();
            for(int i=0; i<data.numAttributes(); i++) {
                atts.add(data.attribute(i));
            }
            List<Example> unkMem = new ArrayList<>();
            start = System.currentTimeMillis();
            for(int i=0, h=0; i<data.size(); i++, h++) {
                Instance ins = data.get(i);
                Example exemplo = new Example(ins.toDoubleArray(), true);

                if(exc.containsKey(exemplo.getRotuloVerdadeiro())) {
                    exc.replace(exemplo.getRotuloVerdadeiro(), exc.get(exemplo.getRotuloVerdadeiro()) + 1);
                } else {
                    exc.put(exemplo.getRotuloVerdadeiro(), 1);
                }

                double rotulo = comite.classify(ins);
                exemplo.setRotuloClassificado(rotulo);
                if(exemplo.getRotuloVerdadeiro() == 3) {
                    ncTest++;
                }
                n++;
                if(rotulo == exemplo.getRotuloVerdadeiro()) {
                    acertos++;
                    acertosTotal++;
                    classifiedExamples++;
                } else if (rotulo == -1) {
                    if(unkRi.containsKey(exemplo.getRotuloVerdadeiro())) {
                        unkRi.replace(exemplo.getRotuloVerdadeiro(), unkRi.get(exemplo.getRotuloVerdadeiro()) + 1);
                    } else {
                        unkRi.put(exemplo.getRotuloVerdadeiro(),1);
                    }
                    unkMem.add(exemplo);
                    if(unkMem.size() >= T) {
                        unkMem = this.newBinaryNoveltyDetection(unkMem, kShort, phi, T);
                    }
                } else {
                    if(!ensemble.knowLabels.contains(rotulo)) {
                        this.fn++;
                        errosTotal++;
                    } else {
                        fe++;
                        erros++;
                        errosTotal++;
                    }
                    classifiedExamples++;
                }

                if(h == 1000) {
                    ClassicMeasures cM = new ClassicMeasures();
                    cM.calcMeasures(fn, fp, fe, nc, classifiedExamples, unkRi, exc);
                    unkRi.clear();
                    exc.clear();
                    fp = 0;
                    fn = 0;
                    fe = 0;
                    nc = 0;
                    h=0;
                    classifiedExamples = 0;
                    classicMeasuresArrayList.add(cM);
                }
            }
        } catch (Exception ex) {
            System.out.println(ex);
            System.out.println(ex.getStackTrace());
        }
        ClassicMeasures cM = new ClassicMeasures();
        cM.calcMeasures(fn, fp, fe, nc, classifiedExamples, unkRi, exc);
        classicMeasuresArrayList.add(cM);
        long elapsed = System.currentTimeMillis() - start;
        double timeInSeconds = Double.parseDouble(String.valueOf(elapsed))/1000;
        System.out.println("Acertos: " + acertosTotal + ", Erros: " + errosTotal);
        HandlesFiles.salvaPredicoes(classicMeasuresArrayList, dataset, nExecution, ensemble.timeOffline, timeInSeconds, acertosTotal, errosTotal);
    }

    private List<Example> newBinaryNoveltyDetection(List<Example> listaDesconhecidos, int kCurto, double phi, int T) {
        int minWeight = T/4;
        FuzzyKMeansClusterer clusters = FuzzyFunctions.fuzzyCMeans(listaDesconhecidos, kCurto, this.ensemble.fuzzification);
        List<CentroidCluster> centroides = clusters.getClusters();
        List<Double> silhuetas = FuzzyFunctions.fuzzySilhouette(clusters, listaDesconhecidos, this.ensemble.alpha);
        List<Integer> silhuetasValidas = new ArrayList<>();

        for(int i=0; i<silhuetas.size(); i++) {
            if(silhuetas.get(i) > 0 && centroides.get(i).getPoints().size() >= minWeight) {
                silhuetasValidas.add(i);
            }
        }

        List<SPFMiC> sfMiCS = FuzzyFunctions.newSeparateExamplesByClusterClassifiedByFuzzyCMeans(listaDesconhecidos, clusters, -1, this.ensemble.alpha, this.ensemble.theta, minWeight);
        List<SPFMiC> sfmicsConhecidos = ensemble.getAllSPFMiCs();
        List<Double> frs = new ArrayList<>();
        List<List<Double>> frsList = new ArrayList<>();


        try {
            for (int i = 0; i < sfMiCS.size(); i++) {
                if (!sfMiCS.get(i).isNull()) {
                    frs.clear();
                    double dist2 = Double.MAX_VALUE;
                    SPFMiC spfMiCMenorDistancia = new SPFMiC();
                    for (int j = 0; j < sfmicsConhecidos.size(); j++) {
                        double dist3 = DistanceMeasures.calculaDistanciaEuclidiana(sfMiCS.get(i).getCentroide(), sfmicsConhecidos.get(j).getCentroide());
                        if (dist3 < dist2) {
                            dist2 = dist3;
                            spfMiCMenorDistancia = sfmicsConhecidos.get(j);
                        }

                        double di = sfmicsConhecidos.get(j).getRadius();
                        double dj = sfMiCS.get(i).getRadius();
                        double dist = (di + dj) / DistanceMeasures.calculaDistanciaEuclidiana(sfmicsConhecidos.get(j).getCentroide(), sfMiCS.get(i).getCentroide());
                        frs.add((di + dj) / dist);
                    }

                    Double minFr = Collections.min(frs);
                    int indexMinFr = frs.indexOf(minFr);
                    if (minFr <= phi) {
                        sfMiCS.get(i).minFr = minFr;
                        sfMiCS.get(i).rotuloMenorDistancia = spfMiCMenorDistancia.getRotulo();
                        sfMiCS.get(i).setRotulo(sfmicsConhecidos.get(indexMinFr).getRotulo());
                    } else {
                        sfMiCS.get(i).minFr = minFr;
                        sfMiCS.get(i).rotuloMenorDistancia = spfMiCMenorDistancia.getRotulo();
                        sfMiCS.get(i).setRotulo(-2);

                    }
                    frsList.add(frs);
                }
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

        for(int i=0; i<centroides.size(); i++) {
            if(silhuetasValidas.contains(i)) {
                List<Example> examplesOfCluster = centroides.get(i).getPoints();
                for(int j=0; j<examplesOfCluster.size(); j++) {
                    examplesOfCluster.get(j).setRotuloClassificado(sfMiCS.get(i).getRotulo());
                    novidadesClassificadas++;
                    classifiedExamples++;
                    if (this.ensemble.knowLabels.contains(examplesOfCluster.get(j).getRotuloVerdadeiro()) && examplesOfCluster.get(j).getRotuloClassificado() == -2) {
                        fp++;
                        erros++;
                        errosTotal++;
                    } else if (this.ensemble.knowLabels.contains(examplesOfCluster.get(j).getRotuloVerdadeiro()) && examplesOfCluster.get(j).getRotuloClassificado() != examplesOfCluster.get(j).getRotuloVerdadeiro()) {
                        fe++;
                        errosTotal++;
                        erros++;
                    } else if (!this.ensemble.knowLabels.contains(examplesOfCluster.get(j).getRotuloVerdadeiro()) && examplesOfCluster.get(j).getRotuloClassificado() != -2) {
                        fn++;
                        errosTotal++;
                        erros++;
                    } else {
                        if(examplesOfCluster.get(j).getRotuloClassificado() == -2) {
                            nc++;
                            ncTotal++;
                        }
                        acertosTotal++;
                    }
                    listaDesconhecidos.remove(examplesOfCluster.get(j));
                }
            }
        }

        return listaDesconhecidos;
    }
}
