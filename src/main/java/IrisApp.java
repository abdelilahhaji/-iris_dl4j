import org.bytedeco.javacpp.Loader;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.nativeblas.Nd4jCpu;

import java.io.File;
import java.io.IOException;

public class IrisApp {
    public static void main(String[] args) throws Exception {
        double learningRate=0.001;
        int numInputs=4;
        int numHidden=10;
        int numOutputs=3;
        System.out.println("Creation du modele");
        MultiLayerConfiguration conf=new NeuralNetConfiguration.Builder()
                .seed(1234)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHidden)
                        .activation(Activation.SIGMOID)
                        .build()
                )
                .layer(1,new OutputLayer.Builder()
                        .nIn(numHidden)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .build()
                )
                .build();
        MultiLayerNetwork model=new MultiLayerNetwork(conf);
        model.init();
        /*Démarrage du serveur de monitoring du processus d'apprentissage */
        UIServer uiServer=UIServer.getInstance();
        InMemoryStatsStorage inMemoryStatsStorage=new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);
        model.setListeners(new StatsListener(inMemoryStatsStorage));
        //System.out.println(conf.toJson());

        System.out.println("Entrainement du modele");
        File fileTrain=new ClassPathResource("iris_training.csv").getFile();
        RecordReader recordReaderTrain=new CSVRecordReader();
        recordReaderTrain.initialize(new FileSplit(fileTrain));
        recordReaderTrain.initialize(new FileSplit(fileTrain));
        int batchSize=1;
        int classIndex=4;
        DataSetIterator dataSetIteratorTrain=
                new RecordReaderDataSetIterator(recordReaderTrain,batchSize,classIndex,numOutputs);
        /*
        while (dataSetIteratorTrain.hasNext()){
            DataSet dataSet=dataSetIteratorTrain.next();
            System.out.println(dataSet.getFeatures());
            System.out.println(dataSet.getLabels());
        }
        */
        int numEpochs=80;
        for (int i = 0; i <numEpochs ; i++) {
            model.fit(dataSetIteratorTrain);
        }

        System.out.println("Evaluation du modele");
        File fileTest=new ClassPathResource("iris_test.csv").getFile();
        RecordReader recordReaderTest=new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(fileTest));
        DataSetIterator dataSetIteratorTest=
                new RecordReaderDataSetIterator(recordReaderTest,batchSize,classIndex,numOutputs);
        Evaluation evaluation=new Evaluation(numOutputs);
        while (dataSetIteratorTest.hasNext()){
            DataSet dataSet = dataSetIteratorTest.next();
            INDArray features=dataSet.getFeatures();
            INDArray targetLabels=dataSet.getLabels();
            INDArray predictedLabels=model.output(features);
            evaluation.eval(predictedLabels,targetLabels);
        }
        System.out.println(evaluation.stats());
        ModelSerializer.writeModel(model,"irisModel.zip",true);

    }
}
