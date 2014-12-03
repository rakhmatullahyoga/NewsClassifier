/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Beans;

import java.io.Serializable;
import custom_weka.CustomWEKA;
import custom_weka.MySQL;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.sql.Date;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Stack;
import javax.faces.bean.ManagedBean;
import javax.faces.bean.RequestScoped;
import javax.faces.bean.SessionScoped;
import javax.faces.context.FacesContext;
import javax.servlet.http.Part;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.WordTokenizer;
import weka.experiment.InstanceQuery;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 *
 * @author luthfi
 */
@ManagedBean(name="Categorizer")
@SessionScoped
public class Categorizer implements Serializable {
    private String title;
    private String article;
    private String category;
    private int category_id;
    private CustomWEKA test;
    private Part part;
    
    public Categorizer()
    {
        title = "";
        article = "";
        category = "";
        test = new CustomWEKA();
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getArticle() {
        return article;
    }

    public void setArticle(String article) {
        this.article = article;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public int getCategory_id() {
        return category_id;
    }

    public void setCategory_id(int category_id) {
        this.category_id = category_id;
    }

    public Part getPart() {
        return part;
    }

    public void setPart(Part part) {
        this.part = part;
    }
    
    
    
    public String newModel() throws SQLException, Exception
    {
        FacesContext facesContext = FacesContext.getCurrentInstance();
        String BasePath = facesContext.getExternalContext().getRealPath("");
        File file = new File(BasePath + File.separator + "dataset/preprocessed.arff");

    	/* This logic is to create the file if the
    	 * file is not already present
    	 */
    	if(!file.exists()){
    	   file.createNewFile();
    	}

    	//Here true is to append the content to file
    	FileWriter fw = new FileWriter(file,true);
    	//BufferedWriter writer give better performance
    	BufferedWriter bw = new BufferedWriter(fw);
    	bw.write("'"+title+"','"+article+"','"+category+"'\n");
    	//Closing BufferedWriter Stream
    	bw.close();
        
        Instances processed_nom = test.ReadDataset(BasePath + File.separator + "dataset/preprocessed.arff");
        NaiveBayesMultinomial nBayes = new NaiveBayesMultinomial();
        test.CreateAndSaveModel(nBayes, processed_nom,BasePath + File.separator);
        
        title = "";
        article = "";
        category = "";
        part = null;
        return "added";
    }
    
    public String reset()
    {
        title = "";
        article = "";
        category = "";
        part = null;
        return "reset";
    }
    
    public String uploadFile() throws IOException, Exception {

        // Extract file name from content-disposition header of file part
        
        FacesContext facesContext = FacesContext.getCurrentInstance();
        
        String basePath = facesContext.getExternalContext().getRealPath("");
        File outputFilePath = new File(basePath + File.separator + "dataset" + File.separator + "template_csv.csv");

        // Copy uploaded file to destination path
        InputStream inputStream = null;
        OutputStream outputStream = null;
        try {
            inputStream = part.getInputStream();
            outputStream = new FileOutputStream(outputFilePath);

            int read = 0;
            final byte[] bytes = new byte[1024];
            while ((read = inputStream.read(bytes)) != -1) {
                outputStream.write(bytes, 0, read);
            }

            //statusMessage = "File upload successfull !!";
        } catch (IOException e) {
            e.printStackTrace();
            //statusMessage = "File upload failed !!";
        } finally {
            if (outputStream != null) {
                outputStream.close();
            }
            if (inputStream != null) {
                inputStream.close();
            }
        }
        
        test.SetModel(basePath + File.separator + "model" + File.separator + "FilteredClassifier.model");
        test.SetUnlabeled(test.ReadFromCSV(basePath + File.separator + "dataset" + File.separator + "template_csv.csv", "14", "6,3"));
        test.SetLabeled(test.ClassifyUnlabeled());
        DataSink.write(basePath + File.separator + "dataset" + File.separator + "NewsLabeled_new_csv.csv", test.GetLabeled());
        return "upload";    // return to same page
    }
    
    public String categorize() throws Exception
    {
        /*String labeledQuerry = "SELECT artikel.JUDUL, artikel.FULL_TEXT, kategori.LABEL "
                + "FROM (artikel NATURAL JOIN artikel_kategori_verified), kategori "
                + "WHERE artikel.ID_ARTIKEL=artikel_kategori_verified.ID_ARTIKEL "
                + "AND kategori.ID_KELAS=artikel_kategori_verified.ID_KELAS;";
        Instances nom = new Instances(test.ReadfromDatabase(labeledQuerry));
        Instances processed_nom = new Instances(test.Preprocess(nom));*/
        FacesContext facesContext = FacesContext.getCurrentInstance();
        String BasePath = facesContext.getExternalContext().getRealPath("");

        // Membuat model dan menyimpannya, kemudian ditrain
        //Instances processed_nom = test.ReadDataset(BasePath + File.separator + "dataset/preprocessed.arff");
        //NaiveBayesMultinomial nBayes = new NaiveBayesMultinomial();
        //test.CreateAndSaveModel(nBayes, processed_nom,BasePath + File.separator);
        
        // Membaca model yang telah disimpan pada file eksternal
        test.SetModel(BasePath + File.separator + "model/FilteredClassifier.model");
        /* Memasukan data ke file */
        PrintWriter writer = new PrintWriter(BasePath + File.separator + "dataset" + File.separator + "unlabeled.arff", "UTF-8");
        writer.println("@relation 'QueryResult-weka.filters.unsupervised.attribute.NominalToString-C1,2'");
        writer.println();
        writer.println("@attribute JUDUL string");
        writer.println("@attribute FULL_TEXT string");
        writer.println("@attribute LABEL {Pendidikan,Politik,'Hukum dan Kriminal','Sosial Budaya',Olahraga,'Teknologi dan Sains',Hiburan,'Bisnis dan Ekonomi',Kesehatan,'Bencana dan Kecelakaan'}");
        writer.println();
        writer.println("@data");
        writer.println("'" + title + "','" + article + "',?");
        writer.close();
        /* Mengklasifikasikan data yang tidak berlabel */
        test.SetUnlabeled(test.ReadDataset(BasePath + File.separator + "dataset" + File.separator + "unlabeled.arff"));
        test.SetLabeled(test.ClassifyUnlabeled());
        
        /* Output hasil klasifikasi */
        DataSink.write(BasePath + File.separator + "dataset" + File.separator + "NewsLabeled.arff", test.GetLabeled());
        FileReader fr = new FileReader(BasePath + File.separator + "dataset" + File.separator + "NewsLabeled.arff");
        BufferedReader textReader = new BufferedReader(fr);
        String readL;
        Stack<String> DataFile = new Stack();
        while ((readL = textReader.readLine()) != null)
        {
            DataFile.add(readL);
        }
        textReader.close();
        category = DataFile.lastElement().split("','")[2].replace("'","");
        return "true";
    }
}
