
import cc.mallet.classify.Classification;
import cc.mallet.classify.Classifier;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.Instance;
import cc.mallet.types.Label;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.LabelVector;
import ca.uwo.csd.ai.nlp.common.SparseVector;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import ca.uwo.csd.ai.nlp.libsvm.ex.SVMPredictor;
import ca.uwo.csd.ai.nlp.libsvm.svm_model;

/**
 * A wrapper for LibSVM classifier.
 * @author Syeed Ibn Faiz
 */
public class SVMClassifier extends Classifier {

    private svm_model model;
    private Map<Label, Double> mltLabel2svmLabel;       //mapping from Mallet to SVM label
    private Map<Double, Label> svmLabel2mltLabel;       //mapping from SVM label to Mallet Label
    private int[] svmIndex2mltIndex;                    //mapping from SVM Label indices (svm.label) to Mallet Label indices (targetLabelAlphabet)
    private boolean predictProbability;                 //whether probability is predicted by SVM

    public SVMClassifier(svm_model model, Map<Label, Double> mLabel2sLabel, Pipe instancePipe, boolean predictProbability) {
        super(instancePipe);
        this.model = model;
        this.mltLabel2svmLabel = mLabel2sLabel;
        this.predictProbability = predictProbability;
        init();        
    }

    private void init() {
        svmLabel2mltLabel = new HashMap<Double, Label>();
        for (Entry<Label, Double> entry : mltLabel2svmLabel.entrySet()) {
            svmLabel2mltLabel.put(entry.getValue(), entry.getKey());
        }

        svmIndex2mltIndex = new int[model.nr_class + 1];
        int[] sLabels = model.label;
        LabelAlphabet labelAlphabet = getLabelAlphabet();
        for (int sIndex = 0; sIndex < sLabels.length; sIndex++) {
            double sLabel = sLabels[sIndex];
            Label mLabel = svmLabel2mltLabel.get(sLabel * 1.0);
            int mIndex = labelAlphabet.lookupIndex(mLabel.toString(), false);
            svmIndex2mltIndex[sIndex] = mIndex;
        }
    }

    @Override
    public Classification classify(Instance instance) {
        SparseVector vector = SVMClassifierTrainer.getVector(instance);
        double[] scores = new double[model.nr_class];
        double sLabel = mltLabel2svmLabel.get(getLabelAlphabet().lookupLabel(instance.getTarget().toString()));
        double p = SVMPredictor.predictProbability(new ca.uwo.csd.ai.nlp.libsvm.ex.Instance(sLabel, vector), model, scores);
        
        //if SVM is not predicting probability then assign a score of 1.0 to the best class(p)
        //and 0.0 to the other classes
        if (!predictProbability) {
            Label label = svmLabel2mltLabel.get(p);
            int index = getLabelAlphabet().lookupIndex(label.toString(), false);
            scores[index] = 1.0;
        } else {
            rearrangeScores(scores);
        }        
        Classification classification = new Classification(instance, this,
                new LabelVector(getLabelAlphabet(), scores));

        return classification;
    }

    /**
     * SVM model's label indices differ from labelAlphabet's label indices, which is why we
     * need to rearrange the score vector returned by the SVM model.
     * @param scores 
     */
    private void rearrangeScores(double[] scores) {
        for (int i = 0; i < scores.length; i++) {
            int mIndex = svmIndex2mltIndex[i];
            double tmp = scores[i];
            scores[i] = scores[mIndex];
            scores[mIndex] = tmp;
        }
    }
}
