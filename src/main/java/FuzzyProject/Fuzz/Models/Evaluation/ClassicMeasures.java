package FuzzyProject.Fuzz.Models.Evaluation;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class ClassicMeasures {
    private double fnew;
    private double mnew;
    private double err;
    private double accuracy;
    private double unkR;
    private double timeOffline;
    private double timeExecution;

    public ClassicMeasures() {

    }

    public double getFnew() {
        return fnew;
    }

    public void setFnew(double fnew) {
        this.fnew = fnew;
    }

    public double getMnew() {
        return mnew;
    }

    public void setMnew(double mnew) {
        this.mnew = mnew;
    }

    public double getErr() {
        return err;
    }

    public void setErr(double err) {
        this.err = err;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
    }

    public double getTimeOffline() {
        return timeOffline;
    }

    public void setTimeOffline(double timeOffline) {
        this.timeOffline = timeOffline;
    }

    public double getTimeExecution() {
        return timeExecution;
    }

    public void setTimeExecution(double timeExecution) {
        this.timeExecution = timeExecution;
    }

    public double getUnkR() {
        return unkR;
    }

    public void setUnkR(double unkR) {
        this.unkR = unkR;
    }

    public void calcMeasures(int fn, int fp, int fe, int nc, int n, Map<Double, Integer> unki, Map<Double, Integer> exci) {
        if(nc == 0) {
            nc = 1;
        }
        this.calculaUnkR(unki, exci);
        this.mnew = (fn*100)/Double.parseDouble(String.valueOf(nc));
        this.fnew = ((fp*100)/(Double.parseDouble(String.valueOf(n))-nc));
        this.err = ((fp+fn+fe)*100)/Double.parseDouble(String.valueOf(n));
        this.accuracy = Double.parseDouble(String.valueOf(100)) - err;
    }

    private void calculaUnkR(Map<Double, Integer> unki, Map<Double, Integer> exci) {
        List<Double> rotulos = new ArrayList<>();
        rotulos.addAll(unki.keySet());
        double unkR = 0;
        for(int i=0; i< unki.size(); i++) {
            double unk = unki.get(rotulos.get(i));
            double exc = exci.get(rotulos.get(i));
            unkR += (unk/exc);
        }
        this.unkR = (unkR/ exci.size()) * 100;
    }
}
