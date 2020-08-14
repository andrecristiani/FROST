package FuzzyProject.Fuzz;

import FuzzyProject.Fuzz.Models.Ensemble;
import FuzzyProject.Fuzz.Models.*;
import FuzzyProject.Fuzz.Models.Evaluation.AcuraciaMedidas;
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

import java.util.*;

public class OnlinePhase {

    public List<Example> exemplosEsperandoTempo = new ArrayList<>();
    public List<ClassicMeasures> desempenho = new ArrayList<>();
    int acertos = 0;
    int acertosTotal = 0;
    int erros = 0;
    int errosTotal = 0;
    Ensemble ensemble;

    int novidadesClassificadas = 0;
    int exemplosClassificados = 0;
    public int fp = 0;
    public int fn = 0;
    public int fe = 0;
    public int nc = 0;
    public int n = 0;
    public int erro1 = 0;
    public int erro2 = 0;
    public int erro3 = 0;
    public int erro4 = 0;
    public int erro5 = 0;
    public int erro6 = 0;

    public int acerto1 = 0;
    public int acerto2 = 0;
    public int acerto3 = 0;
    public int acerto4 = 0;
    public int acerto5 = 0;
    public int acerto6 = 0;

    public int desconhecidos1 = 0;
    public int desconhecidos2 = 0;
    public int desconhecidos3 = 0;
    public int desconhecidos4 = 0;
    public int desconhecidos5 = 0;

    public ClassicMeasures initialize(String caminho, String dataset, Ensemble comite, int latencia, int tChunk, int T, int kShort, double phi) {
        ClassicMeasures c = null;
        List<AcuraciaMedidas> acuracias = new ArrayList<>();
        this.ensemble = comite;
        DataSource source;
        Instances data;
        Instances esperandoTempo;
        int nExeTemp = 0;
        try {
            source = new DataSource(caminho + dataset + "-instances.arff");
            data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            ArrayList<Attribute> atts = new ArrayList<>();
            for(int i=0; i<data.numAttributes(); i++) {
                atts.add(data.attribute(i));
            }
            esperandoTempo = source.getDataSet();
            List<Example> labeledMem = new ArrayList<>();
            List<Example> unkMem = new ArrayList<>();

            int desconhecido = 0;
            for(int i=0, j=0, h=0; i<data.size(); i++, j++, h++) {
//                if(i==25000) {
                    System.out.println(i);
//                }
                Instance ins = data.get(i);
                Example exemplo = new Example(ins.toDoubleArray(), true);
                double rotulo = comite.classify(ins);
                exemplo.setRotuloClassificado(rotulo);
                if(exemplo.getRotuloVerdadeiro() == 3) {
                    nc++;
                }
                n++;
                if(rotulo == exemplo.getRotuloVerdadeiro()) {
                    acertos++;
                    acertosTotal++;
                    if(exemplo.getRotuloClassificado() == 0) {
                        acerto1++;
                    }
                    if(exemplo.getRotuloClassificado() == 1) {
                        acerto2++;
                    }
                    if(exemplo.getRotuloClassificado() == 2) {
                        acerto3++;
                    }
                    if(exemplo.getRotuloClassificado() == 3) {
                        acerto4++;
                    }
                    if(exemplo.getRotuloClassificado() == 4) {
                        acerto5++;
                    }
                } else if (rotulo == -1) {
                    desconhecido++;
                    if(exemplo.getRotuloVerdadeiro() == 0) {
                        desconhecidos1++;
                    }
                    if(exemplo.getRotuloVerdadeiro() == 1) {
                        desconhecidos2++;
                    }
                    if(exemplo.getRotuloVerdadeiro() == 2) {
                        desconhecidos3++;
                    }
                    if(exemplo.getRotuloVerdadeiro() == 3) {
                        desconhecidos4++;
                    }
                    if(exemplo.getRotuloVerdadeiro() == 4) {
                        desconhecidos5++;
                    }
                    unkMem.add(exemplo);
                    if(unkMem.size() >= T) {
                        unkMem = this.newBinaryNoveltyDetection(unkMem, kShort, phi, T);
                    }
                } else {
                    if(exemplo.getRotuloVerdadeiro() == 0) {
                        erro1++;
                        fe++;
                    }
                    if(exemplo.getRotuloVerdadeiro() == 1) {
                        erro2++;
                        fe++;
                    }
                    if(exemplo.getRotuloVerdadeiro() == 2) {
                        erro3++;
                        fe++;
                    }
                    if(exemplo.getRotuloVerdadeiro() == 3) {
                        erro4++;
                        fn++;
                    }
                    if(exemplo.getRotuloVerdadeiro() == 4) {
                        erro5++;
                        fe++;
                    }
                    erros++;
                    errosTotal++;
                }

//                this.exemplosEsperandoTempo.add(exemplo);
//                if(j >= latencia) {
//                    Example labeledExample = new Example(esperandoTempo.get(nExeTemp).toDoubleArray(), true);
//                    labeledMem.add(labeledExample);
//                    if(labeledMem.size() >= tChunk) {
//                        System.err.println("Treinando nova árvore no ponto: " + i);
//                        labeledMem = comite.trainNewClassifier(labeledMem);
//                    }
//                    nExeTemp++;
//                }
            }
//            acuracias.add(Evaluation.calculaAcuracia(acertos, 730, data.size()));
            System.out.println("Acertos: " + acertosTotal);
            System.out.println("Erros: " + errosTotal);
            System.err.println("Erros 1: " + erro1);
            System.err.println("Erros 2: " + erro2);
            System.err.println("Erros 3: " + erro3);
            System.err.println("Erros 4: " + erro4);
            System.err.println("Erros 5: " + erro5);
            System.err.println("Acertos 1: " + acerto1);
            System.err.println("Acertos 2: " + acerto2);
            System.err.println("Acertos 3: " + acerto3);
            System.err.println("Acertos 4: " + acerto4);
            System.err.println("Acertos 5: " + acerto5);
            System.out.println("Desconhecidos: " + desconhecido);
            System.out.println("Sem classificar: " + unkMem.size());
            System.out.println("Desconhecidos1: " + desconhecidos1);
            System.out.println("Desconhecidos2 : " + desconhecidos2);
            System.out.println("Desconhecidos3: " + desconhecidos3);
            System.out.println("Desconhecidos4: " + desconhecidos4);
            System.out.println("Desconhecidos5: " + desconhecidos5);
            c = new ClassicMeasures(fn, fp, fe, nc, n);
            System.out.println("Fnew: " + c.getFnew() + " Mnew: " + c.getMnew() + " Err: " + c.getErr() + " Acurácia: " + c.getAccuracy());
        } catch (Exception ex) {
            System.out.println(ex);
            System.out.println(ex.getStackTrace());
        }
        return c;
    }

    private List<Example> newBinaryNoveltyDetection(List<Example> listaDesconhecidos, int kCurto, double phi, int T) {
//        System.out.println("Executando DN");
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
                    exemplosClassificados++;
                    if (this.ensemble.knowLabels.contains(examplesOfCluster.get(j).getRotuloVerdadeiro()) && examplesOfCluster.get(j).getRotuloClassificado() == -2) {
                    fp++;
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 0) {
                            erro1++;
                        }
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 1) {
                            erro2++;
                        }
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 2) {
                            erro3++;
                        }
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 3) {
                            erro4++;
                        }
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 4) {
//                        System.err.println("Classificado como: " + exemplo.getRotuloClassificado());
                            erro5++;
                        }
                        errosTotal++;
//                      System.err.println("Verdadeiro: " + examplesOfCluster.get(j).getRotuloVerdadeiro() + " classificou como: " + examplesOfCluster.get(j).getRotuloClassificado() + " FR: " + sfMiCS.get(i).minFr + " rótulo menor distância: " + sfMiCS.get(i).rotuloMenorDistancia + " Cluster: " + i);
                    } else if (this.ensemble.knowLabels.contains(examplesOfCluster.get(j).getRotuloVerdadeiro()) && examplesOfCluster.get(j).getRotuloClassificado() != examplesOfCluster.get(j).getRotuloVerdadeiro()) {
                    fe++;
//                    feGlobal++;
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 0) {
                            erro1++;
                        }
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 1) {
                            erro2++;
                        }
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 2) {
                            erro3++;
                        }
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 3) {
                            erro4++;
                        }
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 4) {
//                        System.err.println("Classificado como: " + exemplo.getRotuloClassificado());
                            erro5++;
                        }
                        errosTotal++;
//                        System.err.println("Verdadeiro: " + examplesOfCluster.get(j).getRotuloVerdadeiro() + " classificou como: " + examplesOfCluster.get(j).getRotuloClassificado() + " FR: " + sfMiCS.get(i).minFr + " rótulo menor distância: " + sfMiCS.get(i).rotuloMenorDistancia + " Cluster: " + i);
                    } else {
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 0) {
                            acerto1++;
                        }
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 1) {
                            acerto2++;
                        }
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 2) {
                            acerto3++;
                        }
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 3) {
                            acerto4++;
                        }
                        if(examplesOfCluster.get(j).getRotuloVerdadeiro() == 4) {
                            acerto5++;
                        }
                        acertosTotal++;
                    }
                    listaDesconhecidos.remove(examplesOfCluster.get(j));
                }
            }
        }

        return listaDesconhecidos;
    }

    private List<Example> multiClassNoveltyDetection(List<Example> listaDesconhecidos, int kCurto, double phi, int T) {
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

        for(int i=0; i<sfMiCS.size(); i++) {
            if(!sfMiCS.get(i).isNull()) {
                frs.clear();
                double dist2 = Double.MAX_VALUE;
                SPFMiC spfMiCMenorDistancia = new SPFMiC();
                for (int j = 0; j < sfmicsConhecidos.size(); j++) {
                    double dist3 = DistanceMeasures.calculaDistanciaEuclidiana(sfMiCS.get(i).getCentroide(), sfmicsConhecidos.get(j).getCentroide());
                    if(dist3 < dist2) {
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

        for(int i=0; i<centroides.size(); i++) {
            if(silhuetasValidas.contains(i)) {
                List<Example> examplesOfCluster = centroides.get(i).getPoints();
                for(int j=0; j<examplesOfCluster.size(); j++) {
                    examplesOfCluster.get(j).setRotuloClassificado(sfMiCS.get(i).getRotulo());
                    novidadesClassificadas++;
                    exemplosClassificados++;
                    if (this.ensemble.knowLabels.contains(examplesOfCluster.get(j).getRotuloVerdadeiro()) && examplesOfCluster.get(j).getRotuloClassificado() == -2) {
//                    fp++;
                        errosTotal++;
                        System.err.println("Verdadeiro: " + examplesOfCluster.get(j).getRotuloVerdadeiro() + " classificou como: " + examplesOfCluster.get(j).getRotuloClassificado() + " FR: " + sfMiCS.get(i).minFr + " rótulo menor distância: " + sfMiCS.get(i).rotuloMenorDistancia + " Cluster: " + i);
                    } else if (this.ensemble.knowLabels.contains(examplesOfCluster.get(j).getRotuloVerdadeiro()) && examplesOfCluster.get(j).getRotuloClassificado() != examplesOfCluster.get(j).getRotuloVerdadeiro()) {
//                    fe++;
//                    feGlobal++;
                        errosTotal++;
                        System.err.println("Verdadeiro: " + examplesOfCluster.get(j).getRotuloVerdadeiro() + " classificou como: " + examplesOfCluster.get(j).getRotuloClassificado() + " FR: " + sfMiCS.get(i).minFr + " rótulo menor distância: " + sfMiCS.get(i).rotuloMenorDistancia + " Cluster: " + i);
                    } else {
                        acertosTotal++;
                    }
                    listaDesconhecidos.remove(examplesOfCluster.get(j));
                }
            }
        }

        return listaDesconhecidos;
    }

    private int getIndiceDoMaiorValor(double[] array) {
        int index = 0;
        double maior = -1000000;
        for(int i=0; i<array.length; i++) {
            if(array[i] > maior && array[i] < 1){
                index = i;
                maior = array[i];
            }
        }
        return index;
    }
}
