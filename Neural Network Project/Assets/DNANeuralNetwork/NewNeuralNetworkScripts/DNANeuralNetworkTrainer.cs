using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using DNANeuralNet;
using System.Linq;
using DNAMath;

public class DNANeuralNetworkTrainer : MonoBehaviour
{
    /*
    [Header("Importing From File")]
    [SerializeField] ImageHelper.ImportSettings importSettings;
    [SerializeField] bool loadFromFile;
    [SerializeField] bool newMode;
    int MemoryImageNum = 0;

    */
    [SerializeField] bool newMode;
    [SerializeField] string NeuralNetworkName;

    [Header("ImportImages")]
    [SerializeField] List<string> trainingFolderPaths = new List<string>();
    // [SerializeField] List<string> testingFolderPaths = new List<string>();
    [SerializeField] bool useFolders;
    [SerializeField] bool processImages;
    [SerializeField] bool saveImportedTrainingImages;

    [Header("Image Processing")]
    [SerializeField] int maxCopies;
    [SerializeField] int minCopies;
    [SerializeField] bool whiteBackground;

    [Header("Train Network")]
    [SerializeField] int reshuffleIndex;
    [SerializeField] int numOfEpochs;
    [Range(0, 1)] public float trainingSplit;
    [SerializeField] string errorImagePath;

    [Header("Network Settings")]
    [SerializeField] DNANeuralNetworkSettings networkSettings;

    [Header("Image To Data")]
    List<DNADataPoint> allData = new List<DNADataPoint>(); //Will need to be cleared (I think)
    [SerializeField] int outputNum;

    [Header("UI Stuff")]
    [SerializeField] Button LoadImgsBTN;
    [SerializeField] Button TrainBTN;
    //[SerializeField] Text Log;
    [SerializeField] Text Percent;
    [SerializeField] Slider PercentSlider;
    [SerializeField] GameObject logLine;
    [SerializeField] Transform content;

    MessageLog messageLog = new MessageLog();

    DNADataPoint[] allTrainData;
    DNABatch[] batches;
    DNADataPoint[] evaluateData;
    //DataPoint[,] feedingData;
    DNANeuralNetwork bestNetwork;
    float lastAccuracy;
    double bestCost = 10;
    double currentLearningRate;



    // Start is called before the first frame update
    void Start()
    {
        //LoadImgsBTN.onClick.AddListener(delegate { StartCoroutine(loadImagesFromFile()); });

        // TrainBTN.onClick.AddListener(delegate { StartCoroutine(MemoryTrainNetwork()); });
        StartCoroutine(importFromFolder(trainingFolderPaths, allData));
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public IEnumerator importFromFolder(List<string> paths, List<DNADataPoint> data)
    {

        createLine("Starting Image Importing");
        List<List<Texture2D>> newImages = new List<List<Texture2D>>();
        List<DNADataPoint> evalData = new List<DNADataPoint>();
        int testingIndex = 0;

        for (int i = 0; i < paths.Count; i++)
        {
            //Add new list
            newImages.Add(new List<Texture2D>());

            //Load images
            newImages[i] = Resources.LoadAll<Texture2D>(paths[i]).ToList();

            testingIndex = Mathf.FloorToInt(newImages[i].Count * trainingSplit);

            Debug.Log(newImages[i].Count);

            //Loop through all images
            for (int j = 0; j < newImages[i].Count; j++)
            {
                System.Random rng = new System.Random(Random.Range(0, 1000));
                Texture2D img = newImages[i][j];

                //Debug.Log(RandomInNormalDistribution(rng));

                if (j >= testingIndex)
                {
                    evalData.Add(imageToData(img, i, outputNum));
                }
                else
                {
                    //Check if we process the images
                    if (processImages)
                    {
                        //Maybe remove the thresholds
                        //Process images individually

                        double scale = 1 + RandomInNormalDistribution(rng) * 0.1;

                        img = ApplyScale(img, (float)scale);

                        float angle = (float)RandomInNormalDistribution(rng) * 10;

                        //Apply Rotation
                        img = ApplyRotation(img, angle);

                        //Generate offsetNumbers
                        //Used to be 5
                        int offsetX = Mathf.FloorToInt((float)RandomInNormalDistribution(rng) * (img.width / 10));
                        int offsetY = Mathf.FloorToInt((float)RandomInNormalDistribution(rng) * (img.height / 10));

                        //Apply Offset (max 1/3 width and height)
                        img = ApplyOffset(img, offsetX, offsetY);

                        //Apply Noise
                        img = ApplyNoise(img);
                    }

                    //Convert to Data point
                    data.Add(imageToData(img, i, outputNum));

                }

                Percent.text = (float)j / newImages[i].Count * 100 + " % ";
                PercentSlider.value = (float)j / newImages[i].Count;
                yield return null;

            }

            createLine("Finished Importing " + i);
            yield return null;
        }

        evaluateData = evalData.ToArray();

        StartCoroutine(trainNetwork());

    }

    public IEnumerator trainNetwork()
    {
        System.DateTime trainStart = System.DateTime.UtcNow;

        System.DateTime trainEnd = System.DateTime.UtcNow; ;

        createLine("Starting Network Training");
        yield return null;

        createLine("Network Info: " + "Size:" + sizeToString(networkSettings.networkSize) + "  NumPerBatch: " + networkSettings.dataPerBatch + "  LearnRate: " + networkSettings.initialLearningRate);

        //Create a new Neural Network
        //NeuralNetwork neuro = new NeuralNetwork(networkSettings.neuralNetSize, hiddenActivation, outputActivation, costType);

        DNANeuralNetwork neuro = new DNANeuralNetwork(networkSettings.networkSize, DNAActivation.GetActivationFromType(networkSettings.activationType), DNAActivation.GetActivationFromType(networkSettings.outputActivationType), DNACost.GetCostFromType(networkSettings.costType));

        //Set Cost function
        neuro.SetCostFunction(DNACost.GetCostFromType(networkSettings.costType));

        currentLearningRate = networkSettings.initialLearningRate;

        createLine("Starting Data Shuffle");
        yield return null;

        int total = allData.Count;
        int count = 0;

        System.Random rng = new System.Random();

        List<DNADataPoint> shuffledData;
        int numOfBatches = 0;

        for (int epoch = 0; epoch < numOfEpochs; epoch++)
        {
            if (epoch % reshuffleIndex == 0)
            {

                yield return StartCoroutine(ShuffleArray(allData));

                yield return StartCoroutine(ShuffleArray(evaluateData));

                shuffledData = allData;

                allTrainData = shuffledData.ToArray();

                //Make batches
                //Calculate How many batches needed
                numOfBatches = Mathf.FloorToInt(allTrainData.Length / networkSettings.dataPerBatch);

                createLine("Starting Batching");
                yield return null;

                batches = new DNABatch[numOfBatches];

                for (int i = 0; i < batches.Length; i++)
                {
                    batches[i] = new DNABatch(networkSettings.dataPerBatch);

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
            }

            createLine("Epoch: " + epoch);

            //Shuffle Batches
            yield return StartCoroutine(ShuffleArray(batches));

            // StartCoroutine(displayCost(false, false, neuro, evaluateData));

            // StartCoroutine(displayCost(true, false, neuro, evaluateData));

            //Teaching
            for (int i = 0; i < batches.Length; i++)
            {
                //(1.0 / (1.0 + networkSettings.learnRateDecay * epoch)) *
                neuro.Learn(batches[i].data,  currentLearningRate, networkSettings.regularization, networkSettings.momentum);

                Percent.text = "Teaching: " + (float)i / numOfBatches * 100 + " % ";
                PercentSlider.value = (float)i / numOfBatches;
                yield return null;
            }

            //  StartCoroutine(displayCost(false, true, neuro, evaluateData));

            yield return StartCoroutine(displayCost(true, true, neuro, evaluateData));

            yield return StartCoroutine(EvaluateNetwork(neuro, evaluateData));

        }

        //Once Finished
        createLine("Finished");
       // createLine("Total Time elapsed: " + (trainEnd - startTime));
        yield return null;


    }

    public IEnumerator EvaluateNetwork(DNANeuralNetwork neuro, DNADataPoint[] data)
    {
        int accuracy = 0;

        for (int i = 0; i < data.Length; i++)
        {
            (int, DNAMatrix) classify = neuro.Classify(data[i].inputs);


            if (newMode)
            {
                //0 = minx
                //1 = miny
                //2 = maxx
                //3 = maxy
                Vector2 minGuess = new Vector2((float)classify.Item2[0], (float)classify.Item2[1]);
                Vector2 maxGuess = new Vector2((float)classify.Item2[2], (float)classify.Item2[3]);


                Vector2 min = new Vector2((float)data[i].expectedOutputs[0], (float)data[i].expectedOutputs[1]);
                Vector2 max = new Vector2((float)data[i].expectedOutputs[2], (float)data[i].expectedOutputs[3]);


                Debug.Log("Guess: " + minGuess + " " + maxGuess + "    to    : " + min + " " + max);
            }
            else
            {
                //int classify = neuro.Classify(data[i].inputs);
                int label = data[i].label;

                //Debug.Log(classify.Item1 + " : " + label);

                if (classify.Item1 == label)
                {
                    accuracy++;
                }
                else
                {
                   // dataToImage(data[i], classify.ToString(), errorImagePath, i);
                }
            }

            Percent.text = "Evaluating: " + (float)i / data.Length * 100 + " % ";
            PercentSlider.value = (float)i / data.Length;
            yield return null;
        }



        float actualAccuracy = (float)accuracy / data.Length * 100;

        createLine("Accuracy: " + actualAccuracy + " %");
        Debug.Log("Accuracy: " + actualAccuracy + " %");


        StartCoroutine(saveBestNetwork(neuro, actualAccuracy));

        yield return null;

        //yield return StartCoroutine(saveBestNetwork(neuro, actualAccuracy, data));
    }

   
    public void createLine(string message)
    {
        GameObject line = Instantiate(logLine, content);

        line.transform.GetComponent<Text>().text = message;

        messageLog.addMessage(message);

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

    public IEnumerator ShuffleArray(List<string> data)
    {
        int elementsRemainingToShuffle = data.Count;
        int randomIndex = 0;
        System.Random prng = new System.Random();

        while (elementsRemainingToShuffle > 1)
        {
            // Choose a random element from array
            randomIndex = prng.Next(0, elementsRemainingToShuffle);
            string chosenElement = data[randomIndex];

            // Swap the randomly chosen element with the last unshuffled element in the array
            elementsRemainingToShuffle--;
            data[randomIndex] = data[elementsRemainingToShuffle];
            data[elementsRemainingToShuffle] = chosenElement;

            Percent.text = "Shuffling Data: " + (float)(data.Count - elementsRemainingToShuffle) / data.Count * 100 + " % ";
            PercentSlider.value = (float)(data.Count - elementsRemainingToShuffle) / data.Count;
            yield return null;
        }
    }

    public DNADataPoint imageToData(Texture2D image, int labelIndex, int labelNum)
    {

         DNAMatrix pixels = new DNAMatrix(image.height * image.width, 1);

        for (int x = 0; x < image.width; x++)
        {
            for (int y = 0; y < image.height; y++)
            {
                Color pixelVal = image.GetPixel(x, y);

                double val = (pixelVal.r + pixelVal.g + pixelVal.b) / 3;

                pixels[x * image.width + y] = val;
            }
        }

        DNADataPoint data = new DNADataPoint(pixels, labelIndex, labelNum);

        return data;

    }
    public IEnumerator displayCost(bool all, bool after, DNANeuralNetwork neuro, DNADataPoint[] data)
    {
        if (after)
        {
            if (all)
            {
                double cost = neuro.GetCost(data);
                createLine("All Cost After: " + cost);
                Debug.Log("Cost: " + cost);

                if (cost < bestCost)
                    bestCost = cost;
                else
                    currentLearningRate = currentLearningRate / 2;
                
                yield return null;
            }
            else
            {
                // createLine("Single Cost After: " + neuro.Cost(data[0]));
                yield return null;
            }
        }
        else
        {
            if (all)
            {
                // createLine("All Cost Before: " + neuro.Cost(data));
                yield return null;
            }
            else
            {

                // createLine("Single Cost Before: " + neuro.Cost(data[0]));
                yield return null;
            }
        }
    }

    static double RandomInNormalDistribution(System.Random prng, double mean = 0, double standardDeviation = 1)
    {
        double x1 = 1 - prng.NextDouble();
        double x2 = 1 - prng.NextDouble();

        double y1 = System.Math.Sqrt(-2.0 * System.Math.Log(x1)) * System.Math.Cos(2.0 * System.Math.PI * x2);
        return y1 * standardDeviation + mean;
    }

    public Texture2D ApplyNoise(Texture2D image)
    {
        //5% of pixels get noise


        //Number determines the seed to use
        System.Random rng = new System.Random(Random.Range(0, 100000));

        double noiseProbability = (float)System.Math.Min(rng.NextDouble(), rng.NextDouble()) * 0.05f;
        double noiseStrength = (float)System.Math.Min(rng.NextDouble(), rng.NextDouble());

        Texture2D newImage = image;

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

        return newImage;
    }

    public Texture2D ApplyOffset(Texture2D image, int offsetX, int offsetY)
    {
        Texture2D newImage = new Texture2D(image.width, image.height);

        for (int xIndex = 0; xIndex < newImage.width; xIndex++)
        {
            for (int yIndex = 0; yIndex < newImage.height; yIndex++)
            {

                //Set pixel to the pixel value of the negative offset

                Color col;

                int posX = xIndex - offsetX;
                int posY = yIndex - offsetY;

                bool valid = false;

                if (posX >= 0 && posX <= newImage.width)
                {
                    if (posY >= 0 && posY <= newImage.width)
                    {
                        valid = true;
                    }
                }

                if (valid)
                {
                    col = image.GetPixel(posX, posY);
                }
                else
                {
                    if (whiteBackground)
                    {
                        col = Color.white;
                    }
                    else
                    {
                        col = Color.black;
                    }

                }

                newImage.SetPixel(xIndex, yIndex, col);

            }
        }
        return newImage;


    }

    public Texture2D ApplyScale(Texture2D image, float scaleMult)
    {
        Texture2D newImage = new Texture2D(image.width, image.height);

        for (int xIndex = 0; xIndex < newImage.width; xIndex++)
        {
            for (int yIndex = 0; yIndex < newImage.height; yIndex++)
            {
                int xCenter = Mathf.FloorToInt(image.width / 2);
                int yCenter = Mathf.FloorToInt(image.height / 2);

                int translatedX = xIndex - xCenter;
                int translatedY = yIndex - yCenter;

                //Get radius
                //Divide by multiplier
                float oldRadius = Mathf.Sqrt((translatedX * translatedX) + (translatedY * translatedY)) / scaleMult;

                //Get angle
                float angle = Mathf.Atan2(translatedY, translatedX);

                int oldX = Mathf.FloorToInt(oldRadius * Mathf.Cos(angle));

                int oldY = Mathf.FloorToInt(oldRadius * Mathf.Sin(angle));

                newImage.SetPixel(xIndex, yIndex, getValidPixel(image, oldX, oldY));

            }
        }
        return newImage;
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
            if (whiteBackground)
            {
                return Color.white;
            }
            else
            {
                return Color.black;
            }

        }

    }

    public IEnumerator ShuffleArray(List<DNADataPoint> data)
    {
        int elementsRemainingToShuffle = data.Count;
        int randomIndex = 0;
        System.Random prng = new System.Random();

        while (elementsRemainingToShuffle > 1)
        {
            // Choose a random element from array
            randomIndex = prng.Next(0, elementsRemainingToShuffle);
            DNADataPoint chosenElement = data[randomIndex];

            // Swap the randomly chosen element with the last unshuffled element in the array
            elementsRemainingToShuffle--;
            data[randomIndex] = data[elementsRemainingToShuffle];
            data[elementsRemainingToShuffle] = chosenElement;

            Percent.text = "Shuffling Data: " + (float)(data.Count - elementsRemainingToShuffle) / data.Count * 100 + " % ";
            PercentSlider.value = (float)(data.Count - elementsRemainingToShuffle) / data.Count;
            yield return null;
        }
    }

    public IEnumerator ShuffleArray(DNABatch[] data)
    {
        int elementsRemainingToShuffle = data.Length;
        int randomIndex = 0;
        System.Random prng = new System.Random();

        while (elementsRemainingToShuffle > 1)
        {
            // Choose a random element from array
            randomIndex = prng.Next(0, elementsRemainingToShuffle);
            DNABatch chosenElement = data[randomIndex];

            // Swap the randomly chosen element with the last unshuffled element in the array
            elementsRemainingToShuffle--;
            data[randomIndex] = data[elementsRemainingToShuffle];
            data[elementsRemainingToShuffle] = chosenElement;

            Percent.text = "Shuffling Batches: " + (float)(data.Length - elementsRemainingToShuffle) / data.Length * 100 + " % ";
            PercentSlider.value = (float)(data.Length - elementsRemainingToShuffle) / data.Length;
            yield return null;
        }
    }

    public IEnumerator ShuffleArray(DNADataPoint[] data)
    {
        int elementsRemainingToShuffle = data.Length;
        int randomIndex = 0;
        System.Random prng = new System.Random();

        while (elementsRemainingToShuffle > 1)
        {
            // Choose a random element from array
            randomIndex = prng.Next(0, elementsRemainingToShuffle);
            DNADataPoint chosenElement = data[randomIndex];

            // Swap the randomly chosen element with the last unshuffled element in the array
            elementsRemainingToShuffle--;
            data[randomIndex] = data[elementsRemainingToShuffle];
            data[elementsRemainingToShuffle] = chosenElement;

            Percent.text = "Shuffling Batches: " + (float)(data.Length - elementsRemainingToShuffle) / data.Length * 100 + " % ";
            PercentSlider.value = (float)(data.Length - elementsRemainingToShuffle) / data.Length;
            yield return null;
        }
    }

    public IEnumerator saveBestNetwork(DNANeuralNetwork network, float accuracy)
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
                    //Replace Network witht the new one
                    bestNetwork = network;
                    createLine("New Best Network Made");

                    Debug.Log("New Best Network Made");

                    saveNetwork(bestNetwork, NeuralNetworkName + " (Best)");
                }
                else
                {
                    bestNetwork = network;
                    lastAccuracy = accuracy;

                    createLine("New Best Network Made");
                    Debug.Log("New Best Network Made");

                    saveNetwork(bestNetwork, NeuralNetworkName + " (Best)");

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

    private void saveNetwork(DNANeuralNetwork neuro, string name)
    {
        var dir = "Assets/Resources/NeuralNetworks/MatrixNetworks" + "/" + name + ".json";

        string jsonData = JsonUtility.ToJson(neuro, true);

        Debug.Log(jsonData);

        File.WriteAllText(dir, jsonData);

        createLine("Saved");

        Debug.Log("Saved");
    }

}
