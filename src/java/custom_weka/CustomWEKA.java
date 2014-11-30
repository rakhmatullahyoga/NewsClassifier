/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package custom_weka;

/**
 * @author Rakhmatullah Yoga Sutrisna - 13512053
 */

import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Stopwords;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.InstanceQuery;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class CustomWEKA {
    private Instances labeled;
    private Instances unlabeled;
    private Classifier clasifier;
    
    /**
     * Membaca dataset dari file dataset yang sudah ada ada (format .arff)
     * @param FilePath path lokasi file dataset
     * @return 
     * @throws Exception 
     */
    public Instances ReadDataset(String FilePath) throws Exception {
        Instances dataset = DataSource.read(FilePath);
        dataset.setClassIndex(dataset.numAttributes()-1);
        return dataset;
    }
    /**
     * Membaca dataset dari database
     * @param mQuerry query pemilihan database
     * @return 
     * @throws Exception 
     */
    public Instances ReadfromDatabase(String mQuerry) throws Exception {
        InstanceQuery query;
        Instances nonSTW;
        Instances dataset;
        StringToWordVector strToWV;
        query = new InstanceQuery();
        query.setDatabaseURL("jdbc:mysql://localhost:3306/news_aggregator");
        query.setUsername("root");
        query.setPassword("");
        query.setQuery(mQuerry);
        nonSTW = query.retrieveInstances();
        // Harus diubah dulu tipe atributnya
        NominalToString filter = new NominalToString();
        filter.setAttributeIndexes("1,2");
        filter.setInputFormat(nonSTW);
        dataset = Filter.useFilter(nonSTW, filter);
        strToWV = new StringToWordVector();
        strToWV.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, strToWV);
        Attribute attr = dataset.attribute("LABEL");
        dataset.setClass(attr);
        return dataset;
    }
    /**
     * Training dengan 10Fold Cross Validation
     * @param dataset
     * @throws Exception 
     */
    public void TenFoldTrain(Instances dataset) throws Exception {
        Evaluation eval;
        eval = new Evaluation(dataset);
        eval.crossValidateModel(clasifier, dataset, 10, new Random(1));
        System.out.println(eval.toSummaryString("Results - "+clasifier.getClass().getSimpleName(), false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    /**
     * Full training
     * @param dataset
     * @throws Exception 
     */
    public void FullTraining(Instances dataset) throws Exception {
        Evaluation eval;
        clasifier.buildClassifier(dataset);
        eval = new Evaluation(dataset);
        eval.evaluateModel(clasifier, dataset);
        System.out.println(eval.toSummaryString("Results - "+clasifier.getClass().getSimpleName(), false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    /**
     * Membuat dan menyimpan model hasil pembelajaran
     * @param cls Classifier yang dipilih (J48, kNN, Naive Bayes, Multilayer Perceptron)
     * @param dataset
     * @throws Exception 
     */
    public void CreateAndSaveModel(Classifier cls, Instances dataset) throws Exception {
        clasifier = cls;
        clasifier.buildClassifier(dataset);
        SerializationHelper.write("model/"+cls.getClass().getSimpleName()+".model", cls);
    }
    /**
     * Membaca model yang telah dibuat
     * @param Filepath
     * @throws Exception 
     */
    public void SetModel(String Filepath) throws Exception {
        clasifier = (Classifier) SerializationHelper.read(Filepath);
    }
    /**
     * Mengklasifikasikan dataset yang belum terlabel
     * @return 
     * @throws Exception 
     */
    public Instances ClassifyUnlabeled() throws Exception {
        Instances dataset;
        dataset = new Instances(unlabeled);
        for(int i=0; i<unlabeled.numInstances(); i++) {
            double clsLabel = clasifier.classifyInstance(unlabeled.instance(i));
            dataset.instance(i).setClassValue(clsLabel);
        }
        return dataset;
    }
    
    /* Setter dan Getter */
    public void SetLabeled(Instances _labeled) {
        labeled = _labeled;
    }
    public void SetUnlabeled(Instances _unlabeled) {
        unlabeled = _unlabeled;
    }
    public Instances GetLabeled() {
        return labeled;
    }
    public Instances GetUnlabeled() {
        return unlabeled;
    }
    
    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        Stopwords stpword = new Stopwords();
        
        // Membaca dataset awal
        CustomWEKA test = new CustomWEKA();
        String labeledQuerry = "SELECT artikel.JUDUL, artikel.FULL_TEXT, kategori.LABEL "
                + "FROM (artikel NATURAL JOIN artikel_kategori_verified), kategori "
                + "WHERE artikel.ID_ARTIKEL=artikel_kategori_verified.ID_ARTIKEL "
                + "AND kategori.ID_KELAS=artikel_kategori_verified.ID_KELAS;";
        test.SetLabeled(test.ReadfromDatabase(labeledQuerry));

        // Membuat model dan menyimpannya, kemudian ditrain
        test.CreateAndSaveModel(new J48(), test.GetLabeled());                          // J48
        test.TenFoldTrain(test.GetLabeled());
        test.FullTraining(test.GetLabeled());
        
        test.CreateAndSaveModel(new NaiveBayes(), test.GetLabeled());                   // Naive Bayes
        test.TenFoldTrain(test.GetLabeled());
        test.FullTraining(test.GetLabeled());
        
        test.CreateAndSaveModel(new IBk(), test.GetLabeled());                          // k-NN
        test.TenFoldTrain(test.GetLabeled());
        test.FullTraining(test.GetLabeled());
        
        /* BEWARE, MODEL INI LAMA BANGET, SUMPAH! 
        test.CreateAndSaveModel(new MultilayerPerceptron(), test.GetLabeled());         // Multilayer Perceptron
        test.TenFoldTrain(test.GetLabeled());
        test.FullTraining(test.GetLabeled()); */
        
        /* Mengambil data yang tidak berlabel */
        // querynya belom dibikin buat nyari yang gak berlabel
        /*
        String unlabeledQuerry = "SELECT artikel.JUDUL, artikel.FULL_TEXT, kategori.LABEL "
                + "FROM (artikel NATURAL JOIN artikel_kategori_verified), kategori "
                + "WHERE artikel.ID_ARTIKEL=artikel_kategori_verified.ID_ARTIKEL "
                + "AND kategori.ID_KELAS=artikel_kategori_verified.ID_KELAS;";
        test.SetUnlabeled(test.ReadfromDatabase(unlabeledQuerry));
        */
        // Membaca model yang telah disimpan pada file eksternal
        test.SetModel("model/J48.model");
        test.SetModel("model/NaiveBayes.model");
        test.SetModel("model/IBk.model");
        //test.SetModel("model/MultilayerPerceptron.model");
    
        /* Mengklasifikasikan data yang tidak berlabel */
        //test.SetLabeled(test.ClassifyUnlabeled());
        
        /* Output hasil klasifikasi */
        DataSink.write("dataset/NewsLabeled.arff", test.GetLabeled());
        
    }
}
