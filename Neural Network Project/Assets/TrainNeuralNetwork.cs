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
    [SerializeField] float regularization;
    [SerializeField] float momentum;

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
    [Range(0, 1)] public double noiseProbability;
    [Range(0, 1)] public double noiseStrength;

    List<List<Texture2D>> processedImages = new List<List<Texture2D>>();

    [Header("Image To Data")]
    List<DataPoint> allData = new List<DataPoint>();
    [SerializeField] int outputNum;


    [Header("Train Network")]
    [SerializeField] int numOfEpochs;
    [SerializeField] int costDataDivider; //Remove this?
    [Range(0, 1)] public float trainingSplit;
    //From Asset Folder
    [SerializeField] string exportPath;
    [SerializeField] string errorImagePath;
    [SerializeField] string fileName;

    [SerializeField] bool saveErrorImg;

    DataPoint[] allTrainData;
    Batch[] batches;
    DataPoint[] evaluateData;
    //DataPoint[,] feedingData;
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


    //Feb 24
    //Add scaling to image processing



    // Start is called before the first frame update
    void Start()
    {
        //Create network settings


        networkSettings = new NetworkTrainingInfo(networkSize.neuralNetSize, learnRates, dataPerBatch);

        StartBTN.onClick.AddListener(nextStep);

        //OnDemandRendering.renderFrameInterval = 4;
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

            //Debug.Log(image[i].width);
            //  Debug.Log(image[i].height);

            //  Debug.Log(imgSize.x);
            //   Debug.Log(imgSize.y);

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

        createLine("Starting Data Shuffle");
        yield return null;

        List<DataPoint> shuffledData = new List<DataPoint>();

        int total = allData.Count;
        int count = 0;

        System.Random rng = new System.Random();

        yield return StartCoroutine(ShuffleArray(allData));

        shuffledData = allData;

        /*
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
        */

        //Split up the data into the accuracy/Testing group and the training data
        int accuracyIndex = Mathf.FloorToInt(shuffledData.Count * trainingSplit);

        //Put data in training section
        allTrainData = new DataPoint[accuracyIndex];
        for (int i = 0; i < accuracyIndex; i++)
        {
            allTrainData[i] = shuffledData[i];
        }

        //Get data for evaluating
        evaluateData = new DataPoint[shuffledData.Count - accuracyIndex];
        for (int i = 0; i < evaluateData.Length; i++)
        {
            evaluateData[i] = shuffledData[(allTrainData.Length) + i];
        }

        //Make batches
        //Calculate How many batches needed
        int numOfBatches = Mathf.FloorToInt(allTrainData.Length / networkSettings.dataPerBatch);

        createLine("Starting Batching");
        yield return null;

        batches = new Batch[numOfBatches];

        for (int i = 0; i < batches.Length; i++)
        {
            batches[i] = new Batch(networkSettings.dataPerBatch);

            for (int j = 0; j < networkSettings.dataPerBatch; j++)
            {
                batches[i].addData(allTrainData[networkSettings.dataPerBatch * i + j], j);
            }

            Percent.text = (float)i / numOfBatches * 100 + " % ";
            PercentSlider.value = (float)i / numOfBatches;
            yield return null;
        }

        createLine("Finished Batching");
        yield return null;

        for (int epoch = 0; epoch < numOfEpochs; epoch++)
        {

            createLine("Epoch: " + epoch);

            //Shuffle Batches
            yield return StartCoroutine(ShuffleArray(batches));

            // StartCoroutine(displayCost(false, false, neuro, evaluateData));

            //StartCoroutine(displayCost(true, false, neuro, evaluateData));

            //Teaching
            for (int i = 0; i < batches.Length; i++)
            {
                neuro.Learn(batches[i].data, (1.0 / (1.0 + learnRateDecay * epoch)) * networkSettings.learnRate, regularization, momentum);

                Percent.text = "Teaching: " + (float)i / numOfBatches * 100 + " % ";
                PercentSlider.value = (float)i / numOfBatches;
                yield return null;
            }

            //  StartCoroutine(displayCost(false, true, neuro, evaluateData));

            yield return StartCoroutine(displayCost(true, true, neuro, evaluateData));

            yield return StartCoroutine(EvaluateNetwork(neuro, evaluateData));



            /*
            trainEnd = System.DateTime.UtcNow;

            createLine("Time Elapsed Training: " + (trainEnd - trainStart));
            yield return null;
            */


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

        //Number determines the seed to use
        System.Random rng = new System.Random(Random.Range(0, 100000));

        for (int x = 0; x < image.width; x++)
        {
            for (int y = 0; y < image.height; y++)
            {

                if (rng.NextDouble() <= noiseProbability)
                {
                    double noiseValue = (rng.NextDouble() - 0.5) * noiseStrength;

                    float pixelVal = newImage.GetPixel(x, y).r;

                    pixelVal = System.Math.Clamp(pixelVal - (float)noiseValue, 0, 1);

                    newImage.SetPixel(x, y, new Color(pixelVal, pixelVal, pixelVal));
                }
            }
        }


        /*

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
        */

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
                       
                        saveNetwork(bestNetwork, fileName + " (Best)");
                    }
                }
                else
                {
                    bestNetwork = network;
                    lastAccuracy = accuracy;

                    createLine("New Best Network Made");
                   
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

    public IEnumerator ShuffleArray(List<DataPoint> data)
    {
        int elementsRemainingToShuffle = data.Count;
        int randomIndex = 0;
        System.Random prng = new System.Random();

        while (elementsRemainingToShuffle > 1)
        {
            // Choose a random element from array
            randomIndex = prng.Next(0, elementsRemainingToShuffle);
            DataPoint chosenElement = data[randomIndex];

            // Swap the randomly chosen element with the last unshuffled element in the array
            elementsRemainingToShuffle--;
            data[randomIndex] = data[elementsRemainingToShuffle];
            data[elementsRemainingToShuffle] = chosenElement;

            Percent.text = "Shuffling Data: " + (float)(data.Count - elementsRemainingToShuffle) / data.Count * 100 + " % ";
            PercentSlider.value = (float)(data.Count - elementsRemainingToShuffle) / data.Count;
            yield return null;
        }
    }

    public IEnumerator ShuffleArray(Batch[] data)
    {
        int elementsRemainingToShuffle = data.Length;
        int randomIndex = 0;
        System.Random prng = new System.Random();

        while (elementsRemainingToShuffle > 1)
        {
            // Choose a random element from array
            randomIndex = prng.Next(0, elementsRemainingToShuffle);
            Batch chosenElement = data[randomIndex];

            // Swap the randomly chosen element with the last unshuffled element in the array
            elementsRemainingToShuffle--;
            data[randomIndex] = data[elementsRemainingToShuffle];
            data[elementsRemainingToShuffle] = chosenElement;

            Percent.text = "Shuffling Batches: " + (float)(data.Length - elementsRemainingToShuffle) / data.Length * 100 + " % ";
            PercentSlider.value = (float)(data.Length - elementsRemainingToShuffle) / data.Length;
            yield return null;
        }
    }

    public void dataToImage(DataPoint data, string name, string path, int imgCount)
    {
        //Gte Image size
        //Create texture2D 

        Texture2D img = new Texture2D(subImageSize.x, subImageSize.y);

        for (int x = 0; x < img.width; x++)
        {
            for (int y = 0; y < img.height; y++)
            {
                //Get input index
                //Create color
                float colVal = (float)data.inputs[x * img.height + y];

                Color col = new Color(colVal, colVal, colVal);

                img.SetPixel(x, y, col);
            }
        }

        if (saveErrorImg)
        {
            //Save image
            byte[] bytes = img.EncodeToPNG();
            File.WriteAllBytes(path + "/" + name + " - " + imgCount + ".png", bytes);
        }
    }

    public IEnumerator EvaluateNetwork(NeuralNetwork neuro, DataPoint[] data)
    {
        int accuracy = 0;

        for (int i = 0; i < data.Length; i++)
        {
            int classify = neuro.Classify(data[i].inputs);
            int label = data[i].label;

            Debug.Log(classify + " : " + label);

            if (classify == label)
            {
                accuracy++;
            }
            else
            {
                dataToImage(data[i], classify.ToString(), errorImagePath, i);
            }
            Percent.text = "Evaluating: " + (float)i / data.Length * 100 + " % ";
            PercentSlider.value = (float)i / data.Length;
            yield return null;
        }


        float actualAccuracy = (float)accuracy / data.Length * 100;

        createLine("Accuracy: " + actualAccuracy + " %");
        yield return null;

        yield return StartCoroutine(saveBestNetwork(neuro, actualAccuracy, data));
    }




}


