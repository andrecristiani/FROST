package FuzzyProject.Fuzz.Models.Evaluation;

public class ClassicMeasures {
    private double fnew;
    private double mnew;
    private double err;
    private double accuracy;

    public ClassicMeasures(int fn, int fp, int fe, int nc, int n) {
        this.mnew = (fn*100)/nc;
        this.fnew = ((fp*100)/(n-nc));
        this.err = ((fp+fn+fe)*100)/n;
        this.accuracy = 100 - err;
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
}
