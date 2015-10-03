/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package custom_weka;

/**
 * @author Rakhmatullah Yoga Sutrisna - 13512053
 */

import java.io.File;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.WordTokenizer;
import weka.experiment.InstanceQuery;
import weka.filters.Filter;
import weka.filters.SimpleFilter;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.Remove;
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
     * Membaca dataset dari file CSV
     * @param Path direktori file CSV
     * @param NomIdx indeks atribut yang bertipe nominal (Label)
     * @param StrIdx indeks atribut yang bertipe string (judul, full_text)
     * @return
     * @throws Exception 
     */
    public Instances ReadFromCSV(String Path, String NomIdx, String StrIdx) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(Path));
        loader.setNominalAttributes(NomIdx);
        loader.setStringAttributes(StrIdx);
        Instances data = loader.getDataSet();
        
        Remove remove = new Remove();
        remove.setAttributeIndices(StrIdx+","+NomIdx);
        remove.setInvertSelection(true);
        remove.setInputFormat(data);
        return SimpleFilter.useFilter(data, remove);
    }
    /**
     * Membaca dataset dari database
     * @param mQuerry query pemilihan database
     * @return 
     * @throws Exception 
     */
    public Instances ReadfromDatabase(String mQuerry) throws Exception {
        InstanceQuery query;
        Instances nominal;
        Instances dataset;
        StringToWordVector strToWV;
        query = new InstanceQuery();
        query.setDatabaseURL("jdbc:mysql://localhost:3306/news_aggregator");
        query.setUsername("luthfi");
        query.setPassword("underground");
        query.setQuery(mQuerry);
        nominal = query.retrieveInstances();
        return nominal;
    }
    /**
     * Preproses dataset sebelum membangun model
     * @param nominal dataset yang atributnya masih berupa nominal
     * @return dataset yang atributnya telah dibuat menjadi string
     * @throws Exception 
     */
    public Instances Preprocess(Instances nominal) throws Exception {
        Instances dataset;
        NominalToString filter = new NominalToString();
        filter.setAttributeIndexes("1,2");
        filter.setInputFormat(nominal);
        dataset = Filter.useFilter(nominal, filter);
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
     * @param BasePath
     * @throws Exception 
     */
    public void CreateAndSaveModel(Classifier cls, Instances dataset, String BasePath) throws Exception {
        FilteredClassifier filtercls = new FilteredClassifier();
        StringToWordVector strToWV = new StringToWordVector();
        WordTokenizer token = new WordTokenizer();
        token.setDelimiters("\n \r \t.,;:'\"()?![]1234567890-/");
        strToWV.setAttributeIndices("1,2");
        strToWV.setLowerCaseTokens(true);
        strToWV.setMinTermFreq(5);
        strToWV.setStopwords(new File(BasePath + "stopwords" + File.separator + "stopwordID.txt"));
        strToWV.setTokenizer(token);
        strToWV.setWordsToKeep(1500);
        strToWV.setUseStoplist(true);
        strToWV.setInputFormat(dataset);
        filtercls.setClassifier(cls);
        filtercls.setFilter(strToWV);
        clasifier = filtercls;
        clasifier.buildClassifier(dataset);
        TenFoldTrain(dataset);
        FullTraining(dataset);
        SerializationHelper.write(BasePath + "model" + File.separator+clasifier.getClass().getSimpleName()+".model", clasifier);
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
        Instances classified;
        dataset = new Instances(unlabeled);
        dataset.setClassIndex(dataset.numAttributes()-1);
        classified = new Instances(dataset);
        for(int i=0; i<classified.numInstances(); i++) {
            double clsLabel = clasifier.classifyInstance(dataset.instance(i));
            classified.instance(i).setClassValue(clsLabel);
        }
        return classified;
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
        // Membaca dataset awal
        CustomWEKA test = new CustomWEKA();
        /*String labeledQuerry = "SELECT artikel.JUDUL, artikel.FULL_TEXT, kategori.LABEL "
                + "FROM (artikel NATURAL JOIN artikel_kategori_verified), kategori "
                + "WHERE artikel.ID_ARTIKEL=artikel_kategori_verified.ID_ARTIKEL "
                + "AND kategori.ID_KELAS=artikel_kategori_verified.ID_KELAS;";
        Instances nom = new Instances(test.ReadfromDatabase(labeledQuerry));
        Instances processed_nom = new Instances(test.Preprocess(nom));

        // Membuat model dan menyimpannya, kemudian ditrain
        NaiveBayesMultinomial nBayes = new NaiveBayesMultinomial();
        test.CreateAndSaveModel(nBayes, processed_nom,"");*/
        
        // Membaca model yang telah disimpan pada file eksternal
        test.SetModel("model/FilteredClassifier.model");
        
        /* Mengklasifikasikan data yang tidak berlabel */
        //test.SetUnlabeled(test.ReadDataset("dataset/unlabeled.arff"));
        //DataSink.write("dataset/NewsUnLabeled.csv", test.GetUnlabeled());
        //Instances csv = test.ReadDataset("dataset/NewsUnLabeled.csv");
        
        /*CSVLoader loader = new CSVLoader();
        loader.setSource(new File("dataset/template_csv.csv"));
        loader.setNominalAttributes("14");
        loader.setStringAttributes("3,6");
        Instances data = loader.getDataSet();
        
        Remove remove = new Remove();
        remove.setAttributeIndices("3,6,14");
        remove.setInvertSelection(true);
        remove.setInputFormat(data);*/
        
        test.SetUnlabeled(test.ReadFromCSV("dataset/template_csv.csv", "14", "6,3"));
        //DataSink.write("dataset/daricsv.arff", data);
        test.SetLabeled(test.ClassifyUnlabeled());
        
        
        
        /* Output hasil klasifikasi */
        DataSink.write("dataset/NewsLabeled_new_csv.csv", test.GetLabeled());
        
    }
}
