using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using UnityEngine.Rendering;

public class TrainNeuralNetwork : MonoBehaviour
{
    [Header("Network Settings")]
    [SerializeField] NetworkSize networkSize;
    [SerializeField] int dataPerBatch;
    [SerializeField] float learnRates;
    [SerializeField] float learnRateDecay;

    NetworkTrainingInfo networkSettings;

    [Header("Big To Small Conversion")]

    [SerializeField] List<Texture2D> images = new List<Texture2D>();
    [SerializeField] Vector2Int subImageSize;

    List<List<Texture2D>> miniImages = new List<List<Texture2D>>();

    [Header("Image Processing")]
    [SerializeField] int maxCopies;
    [SerializeField] int minCopies;
    [SerializeField] int maxAngle;
    [SerializeField] int minAngle;

    List<List<Texture2D>> processedImages = new List<List<Texture2D>>();

    [Header("Image To Data")]
    List<DataPoint> allData = new List<DataPoint>();
    [SerializeField] int outputNum;


    [Header("Train Network")]
    [SerializeField] int numOfEpochs;
    [SerializeField] int costDataDivider;
    //From Asset Folder
    [SerializeField] string exportPath;
    [SerializeField] string fileName;

    DataPoint[] allTrainData;
    DataPoint[,] feedingData;
    NeuralNetwork bestNetwork;
    float lastAccuracy;


    [Header("UI Stuff")]
    [SerializeField] Button StartBTN;
    //[SerializeField] Text Log;
    [SerializeField] Text Percent;
    [SerializeField] Slider PercentSlider;
    [SerializeField] GameObject logLine;
    [SerializeField] Transform content;


    int step = 0;

    bool finished = true;
    System.DateTime startTime;
    System.DateTime endTime;

    //string messageLog = "";
    MessageLog messageLog = new MessageLog();

    



    //Feb 23 Implement following features

    //Save the neural Network to the file every time the accuracy increases
    //Remake all the numbers at higher resolution (32x32) for now to see if the images were just too small





    // Start is called before the first frame update
    void Start()
    {
        //Create network settings


        networkSettings = new NetworkTrainingInfo(networkSize.neuralNetSize, learnRates, dataPerBatch);

        StartBTN.onClick.AddListener(nextStep);

        OnDemandRendering.renderFrameInterval = 4;
    }

    // Update is called once per frame
    void Update()
    {

    }



    public void nextStep()
    {

        if (finished)
        {
            startTime = System.DateTime.UtcNow;
            StartCoroutine(createSubImages(images, subImageSize, miniImages));
            finished = false;
        }
        /*
        switch (step)
        {
            case 0:
               
                break;
            case 1:
               

                break;
            case 2:
                

                break;
            case 3:
               
                break;
        }

        step++;
        */
    }

    //
    //Create Images
    //

    public IEnumerator createSubImages(List<Texture2D> image, Vector2Int imgSize, List<List<Texture2D>> newImages)
    {
        System.DateTime subImageStart = System.DateTime.UtcNow;

        createLine("Starting Sub Image Creation");
        yield return null;
        //Start with 2 loops that determine starting coordinates

        int imgCount = 0;

        //Use MultiThreading
        for (int i = 0; i < image.Count; i++)
        {
            newImages.Add(new List<Texture2D>());

            int imgNum = 0;

            Debug.Log(image[i].width);
            Debug.Log(image[i].height);

            Debug.Log(imgSize.x);
            Debug.Log(imgSize.y);

            int totalNum = (image[i].width / imgSize.x) * (image[i].height / imgSize.y);

            for (int startX = 0; startX < image[i].width; startX = startX + imgSize.x)
            {
                for (int startY = 0; startY < image[i].height; startY = startY + imgSize.y)
                {

                    Texture2D newImg = new Texture2D(imgSize.x, imgSize.y, TextureFormat.RGB24, false);

                    for (int pixelX = 0; pixelX < imgSize.x; pixelX++)
                    {
                        for (int pixelY = 0; pixelY < imgSize.y; pixelY++)
                        {
                            newImg.SetPixel(pixelX, pixelY, image[i].GetPixel(startX + pixelX, startY + pixelY));
                        }
                    }

                    newImages[i].Add(newImg);

                    imgCount++;
                    imgNum++;
                    Percent.text = (float)imgNum / totalNum * 100 + " % ";
                    PercentSlider.value = (float)imgNum / totalNum;
                    yield return null;

                }
            }

            createLine(" Finished Image " + i);

            yield return null;

        }

        createLine(imgCount + " Images Created");
        createLine(" Finished Sub Image Creation");

        System.DateTime subImageEnd = System.DateTime.UtcNow;

        createLine("Time Elapsed Creating Sub Images: " + (subImageEnd - subImageStart));
        yield return null;

        //
        //Test for best network settings
        //
        StartCoroutine(createVersions(miniImages, processedImages));

    }

    //
    //Image Processing
    //
    public IEnumerator createVersions(List<List<Texture2D>> inputImages, List<List<Texture2D>> createdImages)
    {
        System.DateTime versionStart = System.DateTime.UtcNow;

        createLine("Starting Image Processing");
        yield return null;
        //Grab all images from the folder
        int allimgCount = 0;

        //Use MultiThreading
        for (int imgType = 0; imgType < inputImages.Count; imgType++)
        {

            createdImages.Add(new List<Texture2D>());

            List<Texture2D> imgs = inputImages[imgType];

            int imgCount = 0;

            for (int j = 0; j < imgs.Count; j++)
            {
                //Number of copies
                int num = Random.Range(minCopies, maxCopies);

                for (int i = 0; i < num; i++)
                {
                    //Copy the image
                    Texture2D newImage = imgs[j];

                    //Get a random rotation 

                    float angle = Random.Range(minAngle, maxAngle);

                    newImage = ApplyRotation(newImage, angle);

                    newImage = ApplyNoise(newImage);


                    createdImages[imgType].Add(newImage);

                    allimgCount++;
                    imgCount++;

                }

                Percent.text = "\n" + (float)j / imgs.Count * 100 + " % ";
                PercentSlider.value = (float)j / imgs.Count;
                yield return null;

            }
            createLine("Finished Layer " + imgType);

            yield return null;

        }

        createLine(allimgCount + " Images Created");
        createLine("Finished Image Processing");

        System.DateTime versionEnd = System.DateTime.UtcNow;

        createLine("Time Elapsed Processing Images: " + (versionEnd - versionStart));
        yield return null;

        StartCoroutine(createData(createdImages, allData));

    }


    //
    //Convert To Data
    //

    public IEnumerator createData(List<List<Texture2D>> procImgs, List<DataPoint> allData)
    {
        System.DateTime dataStart = System.DateTime.UtcNow;

        createLine("Starting Image to Data Conversion");
        yield return null;

        //Use MultiThreading




        for (int type = 0; type < procImgs.Count; type++)
        {

            for (int i = 0; i < procImgs[type].Count; i++)
            {
                allData.Add(imageToData(procImgs[type][i], type, outputNum));

                Percent.text = (float)i / procImgs[type].Count * 100 + " % ";
                PercentSlider.value = (float)i / procImgs[type].Count;
                yield return null;
            }
            createLine("Finished Type " + type);

            yield return null;
        }

        createLine("Finished Converting to Data");
        yield return null;

        System.DateTime dataEnd = System.DateTime.UtcNow;

        createLine("Time Elapsed Converting to Data: " + (dataEnd - dataStart));
        yield return null;

        StartCoroutine(trainNetwork());

    }

    //
    //Train Network
    //

    public IEnumerator trainNetwork()
    {
        System.DateTime trainStart = System.DateTime.UtcNow;

        System.DateTime trainEnd = System.DateTime.UtcNow; ;

        createLine("Starting Network Training");
        yield return null;

        createLine("Network Info: " + "Size:" + sizeToString(networkSettings.neuralNetSize) + "  NumPerBatch: " + networkSettings.dataPerBatch + "  LearnRate: " + networkSettings.learnRate);

        //Create a new Neural Network
        NeuralNetwork neuro = new NeuralNetwork(networkSettings.neuralNetSize);

        //int epoch = 0;


        for (int epoch = 0; epoch < numOfEpochs; epoch++)
        {
            createLine("Starting Data Shuffle");
            yield return null;

            List<DataPoint> shuffledData = new List<DataPoint>();

            int total = allData.Count;
            int count = 0;

            System.Random rng = new System.Random();

            //Shuffle Data
            for (int i = 0; i < total; i++)
            {
                int ranIndex = rng.Next(0, allData.Count - 1);

                // int ranIndex = Random.Range(0, allData.Count - 1);

                shuffledData.Add(allData[ranIndex]);

                allData.RemoveAt(ranIndex);

                count++;

                Percent.text = (float)count / total * 100 + " % ";
                PercentSlider.value = (float)count / total;
                yield return null;

            }

            allData = shuffledData;

            allTrainData = shuffledData.ToArray();


            DataPoint[] costData = new DataPoint[allTrainData.Length / costDataDivider];
            for (int i = 0; i < allTrainData.Length / costDataDivider; i++)
            {
                costData[i] = allTrainData[i];
            }


            //Calculate How many batches needed
            int numOfBatches = Mathf.FloorToInt(allTrainData.Length / networkSettings.dataPerBatch);

            feedingData = new DataPoint[numOfBatches, networkSettings.dataPerBatch];

            createLine("Starting Batching");
            yield return null;

            for (int i = 0; i < numOfBatches; i++)
            {
                for (int j = 0; j < networkSettings.dataPerBatch; j++)
                {
                    feedingData[i, j] = allTrainData[networkSettings.dataPerBatch * i + j];

                }

                Percent.text = (float)i / numOfBatches * 100 + " % ";
                PercentSlider.value = (float)i / numOfBatches;
                yield return null;

            }

            createLine("Finished Batching");
            yield return null;

            //for (int trainIndex = 0; trainIndex < 2; trainIndex++)
            //{
            createLine("Epoch: " + epoch);

            StartCoroutine(displayCost(false, false, neuro, costData));

            StartCoroutine(displayCost(true, false, neuro, costData));

            for (int i = 0; i < numOfBatches; i++)
            {

                DataPoint[] batch = new DataPoint[networkSettings.dataPerBatch];

                for (int j = 0; j < networkSettings.dataPerBatch; j++)
                {

                    batch[j] = feedingData[i, j];

                }

                //
                //Teach the Neural Network
                neuro.Learn(batch, (1.0 / (1.0 + learnRateDecay * epoch)) * networkSettings.learnRate,0, 0);

                Percent.text = (float)i / numOfBatches * 100 + " % ";
                PercentSlider.value = (float)i / numOfBatches;
                yield return null;
            }

            createLine("Finished Teaching");
            yield return null;

            StartCoroutine(displayCost(false, true, neuro, costData));

            StartCoroutine(displayCost(true, true, neuro, costData));

            createLine("Start Testing");
            yield return null;

            List<DataPoint> testing = new List<DataPoint>();

            createLine("Creating Testing Data");
            yield return null;

            for (int i = 0; i < processedImages.Count; i++)
            {
                for (int j = 0; j < networkSettings.dataPerBatch; j++)
                {

                    testing.Add(imageToData(processedImages[i][j], i, networkSettings.neuralNetSize[networkSettings.neuralNetSize.Length - 1]));

                }
            }

            int accuracy = 0;

            createLine("Evaluating");
            yield return null;

            foreach (DataPoint data in shuffledData)
            {
                Debug.Log(neuro.Classify(data.inputs) + " : " + data.label);

                if (neuro.Classify(data.inputs) == data.label)
                {
                    accuracy++;
                }
            }

            float actualAccuracy = (float)accuracy / shuffledData.Count * 100;

            StartCoroutine(saveBestNetwork(neuro, actualAccuracy, costData));

            createLine("Accuracy: " + actualAccuracy + " %");
            yield return null;

            trainEnd = System.DateTime.UtcNow;

            createLine("Time Elapsed Training: " + (trainEnd - trainStart));
            yield return null;


        }

        //Once Finished

        createLine("Total Time elapsed: " + (trainEnd - startTime));
        yield return null;

        saveNetwork(neuro, fileName + " Final");

        /*
        if (saveBest)
        {
            createLine("Best Network has the following stats");
            createLine("Final Accuracy: " + lastAccuracy * 100 + " %");
            StartCoroutine(displayCost(true, true, bestNetwork, allTrainData));

            var direc = "Assets/Resources/NeuralNetworks/BestNetworks" + "/" + fileName + ".json";

            string jsonDat = JsonUtility.ToJson(bestNetwork, true);

            Debug.Log(jsonDat);

            File.WriteAllText(direc, jsonDat);
        }
        */

        saveMessageLog();

        finished = true;

    }





    //
    //
    //Image Processing Shit
    //
    //
    public void ExpandImage(Texture2D image, int scaleFactor, string name)
    {
        //Create a copy image of expanded dimensions


        Debug.Log(image.width);
        Debug.Log(image.width);

        Texture2D newImage = new Texture2D(image.width * scaleFactor, image.height * scaleFactor);


        Debug.Log(newImage.width);
        Debug.Log(newImage.width);

        for (int xIndex = 0; xIndex < newImage.width; xIndex++)
        {
            for (int yIndex = 0; yIndex < newImage.height; yIndex++)
            {

                int xPos = Mathf.FloorToInt(xIndex / scaleFactor);
                int yPos = Mathf.FloorToInt(yIndex / scaleFactor);

                newImage.SetPixel(xIndex, yIndex, image.GetPixel(xPos, yPos));

            }
        }

        //Save Texture as PNG
        byte[] bytes = newImage.EncodeToPNG();


        File.WriteAllBytes("Assets/New Images/Testing/" + name + ".png", bytes);

    }

    public Texture2D ApplyRotation(Texture2D image, float angle)
    {
        float angRad = (Mathf.PI / 180) * angle;

        Texture2D newImage = new Texture2D(image.width, image.height);


        for (int xIndex = 0; xIndex < newImage.width; xIndex++)
        {
            for (int yIndex = 0; yIndex < newImage.height; yIndex++)
            {

                int xCenter = Mathf.FloorToInt(image.width / 2);
                int yCenter = Mathf.FloorToInt(image.height / 2);

                int translatedX = xIndex - xCenter;
                int translatedY = yIndex - yCenter;


                int oldX = Mathf.FloorToInt(translatedX * Mathf.Cos(angRad) - translatedY * Mathf.Sin(angRad));

                int oldY = Mathf.FloorToInt(translatedX * Mathf.Sin(angRad) + translatedY * Mathf.Cos(angRad));

                newImage.SetPixel(xIndex, yIndex, getValidPixel(image, oldX, oldY));

            }
        }
        return newImage;
    }

    public Color getValidPixel(Texture2D image, int oldX, int oldY)
    {
        bool verdict = true;

        int x = oldX + Mathf.FloorToInt(image.width / 2);
        int y = oldY + Mathf.FloorToInt(image.height / 2);

        if ((x < image.width && x >= 0) && (y < image.height && y >= 0))
        {
            //Debug.Log("Hi");
            return image.GetPixel(x, y);
        }
        else
        {
            // Debug.Log("White");
            return Color.white;
        }

    }


    public void Compress(Texture2D image, int scaleFactor, string name)
    {
        Texture2D newImage = new Texture2D(image.width / scaleFactor, image.height / scaleFactor);

        for (int xIndex = 0; xIndex < newImage.width; xIndex++)
        {
            for (int yIndex = 0; yIndex < newImage.height; yIndex++)
            {

                float red = 0;
                float green = 0;
                float blue = 0;

                for (int i = 0; i < scaleFactor; i++)
                {
                    for (int j = 0; j < scaleFactor; j++)
                    {
                        red += image.GetPixel(xIndex * scaleFactor + i, yIndex * scaleFactor + j).r;
                        green += image.GetPixel(xIndex * scaleFactor + i, yIndex * scaleFactor + j).g;
                        blue += image.GetPixel(xIndex * scaleFactor + i, yIndex * scaleFactor + j).b;

                    }
                }

                //Get the averages
                red = red / (scaleFactor * scaleFactor);
                green = green / (scaleFactor * scaleFactor);
                blue = blue / (scaleFactor * scaleFactor);

                newImage.SetPixel(xIndex, yIndex, new Color(red, green, blue));
            }

        }

        //Save Texture as PNG
        byte[] bytes = newImage.EncodeToPNG();


        File.WriteAllBytes("Assets/New Images/Testing/" + name + ".png", bytes);

    }
    public Texture2D ApplyNoise(Texture2D image)
    {
        //5% of pixels get noise

        Texture2D newImage = image;

        float mult = Random.Range(1, 5);

        int num = Mathf.FloorToInt((float)(newImage.width * mult / 5));

        for (int i = 0; i < num; i++)
        {
            int xPos = Mathf.FloorToInt(Random.Range(0, newImage.width));
            int yPos = Mathf.FloorToInt(Random.Range(0, newImage.height));

            float col = (float)Random.Range(0, 255) / 255;

            newImage.SetPixel(xPos, yPos, new Color(col, col, col));

        }

        return newImage;
    }

    //
    //
    //Image to Data
    //
    //

    public DataPoint imageToData(Texture2D image, int labelIndex, int labelNum)
    {

        double[] pixels = new double[image.width * image.height];

        for (int x = 0; x < image.width; x++)
        {
            for (int y = 0; y < image.height; y++)
            {
                Color pixelVal = image.GetPixel(x, y);

                double val = (pixelVal.r + pixelVal.g + pixelVal.b) / 3;

                pixels[x * image.width + y] = val;
            }
        }

        DataPoint data = new DataPoint(pixels, labelIndex, labelNum);

        return data;

    }

    public void createLine(string message)
    {
        GameObject line = Instantiate(logLine, content);

        line.transform.GetChild(0).GetComponent<Text>().text = message;

        messageLog.addMessage(message);

    }

    public IEnumerator displayCost(bool all, bool after, NeuralNetwork neuro, DataPoint[] data)
    {

        if (after)
        {
            if (all)
            {
                createLine("All Cost After: " + neuro.Cost(data));
                yield return null;
            }
            else
            {
                createLine("Single Cost After: " + neuro.Cost(data[0]));
                yield return null;
            }
        }
        else
        {
            if (all)
            {
                createLine("All Cost Before: " + neuro.Cost(data));
                yield return null;
            }
            else
            {

                createLine("Single Cost Before: " + neuro.Cost(data[0]));
                yield return null;
            }
        }


    }

    public void saveMessageLog()
    {
        var dir = "Assets/Resources/NeuralNetworks/MessageLog/" + "MessageLog(" + fileName + ").json";

        string jsonData = JsonUtility.ToJson(messageLog, true);

        Debug.Log(jsonData);

        File.WriteAllText(dir, jsonData);
    }


    public string sizeToString(int[] size)
    {
        string str = "[" + size[0];

        for (int i = 1; i < size.Length - 1; i++)
        {
            str += ", " + size[i];
        }

        str += ", " + size[size.Length - 1] + "]";

        return str;
    }

    public IEnumerator saveBestNetwork(NeuralNetwork network, float accuracy, DataPoint[] allTrainData)
    {
        //Save best
        if (bestNetwork != null)
        {
            //Compare for best

            if ((accuracy) >= lastAccuracy)
            {
                //In the case they are equal, check for the lowest cost
                if (accuracy == lastAccuracy)
                {
                    double bestCost = bestNetwork.Cost(allTrainData);

                    double currentCost = network.Cost(allTrainData);

                    if (currentCost < bestCost)
                    {
                        //Replace Network witht the new one
                        bestNetwork = network;
                        createLine("New Best Network Made");
                        createLine("Accuracy: " + accuracy + " %");
                        createLine("All Cost: " + currentCost);

                        createLine("Saving New Best");
                        saveNetwork(bestNetwork, fileName + " (Best)");
                    }
                }
                else
                {
                    bestNetwork = network;
                    lastAccuracy = accuracy;

                    createLine("New Best Network Made");
                    createLine("Accuracy: " + accuracy + " %");

                    createLine("Saving New Best");
                    saveNetwork(bestNetwork, fileName + " (Best)");

                }

            }
        }
        else
        {
            bestNetwork = network;
            lastAccuracy = accuracy;
        }

        yield return null;
    }

    private void saveNetwork(NeuralNetwork neuro, string name)
    {
        var dir = "Assets/Resources/NeuralNetworks/BestNetworks" + "/" + name + ".json";

        string jsonData = JsonUtility.ToJson(neuro, true);

        Debug.Log(jsonData);

        File.WriteAllText(dir, jsonData);

        createLine("Saved");
    }


}


