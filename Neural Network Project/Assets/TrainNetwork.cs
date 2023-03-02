using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Linq;
using UnityEngine.UI;

public class TrainNetwork : MonoBehaviour
{
    //From Asset Folder
    public string importPath;

    //From Asset Folder
    public string exportPath;

    public string fileName;

    public int[] neuralNetworkSize;

    public int numOfBatches = 100;

    public float learnRate;

    public string testingImportPath;


    [Header("UI Stuff")]
    [SerializeField] Button StartBTN;
    [SerializeField] Text Log;
    [SerializeField] Text Percent;
    [SerializeField] Slider PercentSlider;


    //Neural Related (Training)
    DataStorage data;
    List<DataPoint> allData1;
    List<DataPoint> shuffledData;
    DataPoint[] allData2;
    NeuralNetwork neuro;



    DataStorage testingData;
    List<DataPoint> testingAllDataList;
    List<DataPoint> testingShuffled;
    //DataPoint[] testingAllDataArray;


    ImageToDataPointConverter converter;

    List<DataPoint> testing = new List<DataPoint>();

    public DataPoint[,] feedingData;


    // Start is called before the first frame update
    void Start()
    {

        

        //Alright so Neuro and Neuro 1 should have fully functional classifying 




        converter = Camera.main.GetComponent<ImageToDataPointConverter>();

        //Oh shit, maybe we need to feed the batches with random data? That would most likely help

        //Yeah, I think that will do the trick



        //Load all the datapoints (As List and convert it to array)




       // StartCoroutine(trainNetwork());


        //Oh shit we may have gotten it

        StartBTN.onClick.AddListener(startTraining);


    }

    public IEnumerator trainNetwork ()
    {
        Log.text = "";
        Log.text += "Loading Data";
        yield return null;

        data = loadSaveFromPath(importPath);

        allData1 = data.allData;

        Debug.Log(allData1.Count);

        Log.text += "\n Finished Loading";
        yield return null;

        shuffledData = new List<DataPoint>();

        int total = allData1.Count;
        int count = 0;

        Log.text += "\n Shuffle Data";
        yield return null;

        //Shuffle Data
        while (allData1.Count >= 1)
        {
            int ranIndex = Random.Range(0, allData1.Count - 1);

            shuffledData.Add(allData1[ranIndex]);

            allData1.RemoveAt(ranIndex);

            count++;

            Percent.text = (float)count / total * 100 + " % ";
            PercentSlider.value = (float)count / total;


        }

        Log.text += "\n Shuffling Done";
        yield return null;


        allData2 = shuffledData.ToArray();


        Log.text += "\n Converted To Array";
        yield return null;

        //Create a new Neural Network
        neuro = new NeuralNetwork(neuralNetworkSize, Activation.Sigmoid, Activation.Sigmoid);

        //Calculate cost at first

        int numPerBatch = Mathf.FloorToInt(allData2.Length / numOfBatches);

        feedingData = new DataPoint[numOfBatches, numPerBatch];

        Log.text += "\n Batching";
        yield return null;

        Debug.Log("Batching");

        for (int i = 0; i < numOfBatches; i++)
        {
            for (int j = 0; j < numPerBatch; j++)
            {
                feedingData[i, j] = allData2[numPerBatch * i + j];

            }

        }

        Log.text += "\n Finished Batching";
        yield return null;

        Log.text += "\n Single: " + neuro.Cost(allData2[0]);
        yield return null;

        Log.text += "\n All: " + neuro.Cost(allData2);
        yield return null;


        Log.text += "\n Teaching";
        yield return null;

        for (int i = 0; i < numOfBatches; i++)
        {

            DataPoint[] batch = new DataPoint[numPerBatch];

            for (int j = 0; j < numPerBatch; j++)
            {

                batch[j] = feedingData[i, j];

            }

            //Teach the Neural Network
            neuro.Learn(batch, learnRate);

            Percent.text = (float)count / total * 100 + " % ";
            PercentSlider.value = (float)count / total;
            yield return null;
        }

        Log.text += "\n Finished Teaching";
        yield return null;

        Log.text += "\n Single: " + neuro.Cost(allData2[0]);
        yield return null;

        Log.text += "\n All: " + neuro.Cost(allData2);
        yield return null;

        Log.text += "\n Saving";
        yield return null;

        //Save the Neural Network in a file
        NeuralNetworkSaver saver = new NeuralNetworkSaver(neuro);

        var dir = exportPath + "/" + fileName + ".json";

        string jsonData = JsonUtility.ToJson(saver, true);

        Debug.Log(jsonData);

        File.WriteAllText(dir, jsonData);

        Log.text += "\n Finished Saving";
        yield return null;




        /*

        //
        //Make a Accuracy Tester
        //

        testingData = loadSaveFromPath(testingImportPath);

        yield return null;

        testingAllDataList = data.allData;

        Debug.Log(testingAllDataList.Count);


        testingShuffled = new List<DataPoint>();

        //Shuffle Data
        while (allData1.Count >= 1)
        {
            int ranIndex = Random.Range(0, testingAllDataList.Count - 1);

            testingShuffled.Add(testingAllDataList[ranIndex]);

            testingAllDataList.RemoveAt(ranIndex);

            count++;

            Percent.text = (float)count / total * 100 + " % ";
            PercentSlider.value = (float)count / total;


        }

        yield return null;


        int score = 0;
        for (int i = 0; i < testingShuffled.Count; i ++)
        {
            if (neuro.Classify(testingShuffled[i].inputs) == testingShuffled[i].label)
            {
                score++;
            }
        }

        Log.text += "\n Accuracy: " + (float)score / testingShuffled.Count * 100 + " % ";
        yield return null;

        */


        // 0 = 4
        // 1 = 6


        //Maybe add a validation thing here
        /*
        testing.Add(converter.imageToData(four1, 0, 2));
        testing.Add(converter.imageToData(four2, 0, 2));
        testing.Add(converter.imageToData(four3, 0, 2));

        testing.Add(converter.imageToData(six1, 1, 2));
        testing.Add(converter.imageToData(six2, 1, 2));
        testing.Add(converter.imageToData(six3, 1, 2));



        //Should all be 4
        Debug.Log(neuro.Classify(testing[0].inputs));
        Debug.Log(neuro.Classify(testing[1].inputs));
        Debug.Log(neuro.Classify(testing[2].inputs));

        //Should all be 6
        Debug.Log(neuro.Classify(testing[3].inputs));
        Debug.Log(neuro.Classify(testing[4].inputs));
        Debug.Log(neuro.Classify(testing[5].inputs));

        */


    }

    public void startTraining ()
    {
        StartCoroutine(trainNetwork());
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public DataStorage loadSaveFromPath(string path)
    {
        //This function loads the save named into the currently used save file

        //Debug.Log(path);
        string jsonData = "";
        if (File.Exists(path))
        {
            //Extract JSON Data
            jsonData = File.ReadAllText(path);
            Debug.Log(jsonData);
            return JsonUtility.FromJson<DataStorage>(jsonData);
        }
        else
        {
            Debug.Log("Doesn't exist");
            return null;
        }

    }




}
