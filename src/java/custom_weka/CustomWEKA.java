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
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.InstanceQuery;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class CustomWEKA {
    private static Instances dataset;
    private static Classifier clasifier;
    private static Evaluation eval;
    private static InstanceQuery query;
    
    /**
     * Membaca dataset dari file dataset yang sudah ada ada (format .arff)
     * @param FilePath path lokasi file dataset
     * @throws Exception 
     */
    static void ReadDataset(String FilePath) throws Exception {
        dataset = DataSource.read(FilePath);
        dataset.setClassIndex(dataset.numAttributes()-1);
    }
    /**
     * Membaca dataset dari database
     * @param mQuerry query pemilihan database
     * @throws Exception 
     */
    static void ReadfromDatabase(String mQuerry) throws Exception {
        Instances nonSTW;
        StringToWordVector strToWV;
        query = new InstanceQuery();
        query.setDatabaseURL("jdbc:mysql://localhost:3306/news_aggregator");
        query.setUsername("root");
        query.setPassword("");
        query.setQuery(mQuerry);
        nonSTW = query.retrieveInstances();
        strToWV = new StringToWordVector();
        strToWV.setInputFormat(nonSTW);
        dataset = Filter.useFilter(nonSTW, strToWV);
        dataset.setClassIndex(dataset.numAttributes()-1);
    }
    /**
     * Training dengan 10Fold Cross Validation
     * @throws Exception 
     */
    static void TenFoldTrain() throws Exception {
        eval = new Evaluation(dataset);
        eval.crossValidateModel(clasifier, dataset, 10, new Random(1));
        System.out.println(eval.toSummaryString("Results\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    /**
     * Full training
     * @throws Exception 
     */
    static void FullTraining() throws Exception {
        clasifier.buildClassifier(dataset);
        eval = new Evaluation(dataset);
        eval.evaluateModel(clasifier, dataset);
        System.out.println(eval.toSummaryString("Results\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.fMeasure(1) + " "+eval.precision(1)+" "+eval.recall(1));
        System.out.println(eval.toMatrixString());
    }
    /**
     * Membuat dan menyimpan model hasil pembelajaran
     * @param cls Classifier yang dipilih (J48, kNN, Naive Bayes, Multilayer Perceptron)
     * @throws Exception 
     */
    static void CreateAndSaveModel(Classifier cls) throws Exception {
        clasifier = cls;
        clasifier.buildClassifier(dataset);
        SerializationHelper.write(cls.getClass().getSimpleName()+".model", cls);
    }
    /**
     * Membaca model yang telah dibuat
     * @param Filepath
     * @throws Exception 
     */
    static void ReadModel(String Filepath) throws Exception {
        clasifier = (Classifier) SerializationHelper.read(Filepath);
    }
    /**
     * Mengklasifikasikan dataset yang belum terlabel
     * @param FilePath
     * @throws Exception 
     */
    static void Classify(String FilePath) throws Exception {
        Instances unlabeled = DataSource.read(FilePath);
        unlabeled.setClassIndex(unlabeled.numAttributes()-1);
        Instances labeled = new Instances(unlabeled);
        for(int i=0; i<unlabeled.numInstances(); i++) {
            double clsLabel = clasifier.classifyInstance(unlabeled.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        DataSink.write("newLabeled.arff", labeled);
    }
    
    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {
        // membaca dataset awal
        String mQuerry = "SELECT artikel.JUDUL, artikel.FULL_TEXT, kategori.LABEL FROM (artikel NATURAL JOIN artikel_kategori_verified), kategori WHERE artikel.ID_ARTIKEL=artikel_kategori_verified.ID_ARTIKEL AND kategori.ID_KELAS=artikel_kategori_verified.ID_KELAS;";
        ReadfromDatabase(mQuerry);

        // membuat model dan menyimpannya, kemudian ditrain
        CreateAndSaveModel(new J48());                          // J48
        TenFoldTrain();
        FullTraining();
        
        CreateAndSaveModel(new NaiveBayes());                   // Naive Bayes
        TenFoldTrain();
        FullTraining();
        
        CreateAndSaveModel(new IBk());                          // k-NN
        TenFoldTrain();
        FullTraining();

        /* BEWARE, MODEL INI LAMA BANGET, SUMPAH! 
        CreateAndSaveModel(new MultilayerPerceptron());         // Multilayer Perceptron
        TenFoldTrain();
        FullTraining(); */
        
        
        // membaca model yang telah disimpan pada file eksternal
        /*ReadModel("J48.model");
        ReadModel("NaiveBayes.model");
        ReadModel("IBk.model");
        ReadModel("MultilayerPerceptron.model");*/
    }
}
