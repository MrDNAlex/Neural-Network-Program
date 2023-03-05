using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using UnityEngine.Rendering;
using System.Linq;
using DNANeuralNetwork;

public class TrainNeuralNetwork : MonoBehaviour
{
    [System.Serializable]
    public struct DataFile
    {
        public TextAsset imageFile;
        public TextAsset labelFile;
    }

    [SerializeField] int imageSize = 28;
    [SerializeField] bool greyscale = true;
    [SerializeField] DataFile[] dataFiles;
    [SerializeField] bool loadFromFile;





    [Header("Network Settings")]
    [SerializeField] NeuralNetworkSettings networkSettings;
    /*
    [SerializeField] NetworkSize networkSize;
    [SerializeField] Activation hiddenActivation;
    [SerializeField] Activation outputActivation;
    [SerializeField] Cost costType;
    [SerializeField] int dataPerBatch;
    [SerializeField] float learnRates;
    [SerializeField] float learnRateDecay;
    [SerializeField] float regularization;
    [SerializeField] float momentum;
    */

   // NetworkTrainingInfo networkSettings;

    [Header("Big To Small Conversion")]

    [SerializeField] List<Texture2D> images = new List<Texture2D>();
    [SerializeField] Vector2Int subImageSize;

    List<List<Texture2D>> miniImages = new List<List<Texture2D>>(); //MiniImages need to be cleared

    [Header("ImportImages")]
    [SerializeField] List<string> trainingFolderPaths = new List<string>();
   // [SerializeField] List<string> testingFolderPaths = new List<string>();
    [SerializeField] bool useFolders;
    [SerializeField] bool processImages;
    [SerializeField] bool saveImportedTrainingImages;
   


    [Header("Image Processing")]
    [SerializeField] int maxCopies;
    [SerializeField] int minCopies;
    //[Range(0, 1)] public double noiseProbability;
   // [Range(0, 1)] public double noiseStrength;
    [SerializeField] bool whiteBackground;

    List<List<Texture2D>> processedImages = new List<List<Texture2D>>(); //Processed Images need to be cleared

    [Header("Image To Data")]
    List<DataPoint> allData = new List<DataPoint>(); //Will need to be cleared (I think)
    [SerializeField] int outputNum;


    [Header("Train Network")]
    [SerializeField] int reshuffleIndex;
    [SerializeField] int numOfEpochs;
    [Range(0, 1)] public float trainingSplit;

    //From Asset Folder
    [SerializeField] string exportPath;
    [SerializeField] string errorImagePath;
    [SerializeField] string imageProcessingPath;
    [SerializeField] string fileName;

    [SerializeField] bool saveErrorImg;
    [SerializeField] bool saveImageProcessing;

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


    //Feb 27
    //Recheck the bias equations? There might be an error somewhere in there
    //Add the activation and output activation function 
    //Switch cost function to cross entropy
    //Randomize all settings when creating sub images from now on
    //You know what, it must definitely be overfitting, let's figure out how to download that mnist and extract it, we will save them as individual images and then place them in folders

    //**
  

    
    //Check the image processing settings in the Seb Lague Github
    //When ever you have time, copy over neural network relevant scripts 
    //Copy the image data from seb lague
    //Cross entropy might be broken
    //Fix the angle to maybe *5?


    //Going to have to find a way to make the background of images black

    // Start is called before the first frame update
    void Start()
    {
        //Create network settings

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
            if (loadFromFile)
            {

                StartCoroutine(loadImagesFromFile());

            } else
            {
                if (useFolders)
                {
                    startTime = System.DateTime.UtcNow;

                    //Check if image processing is needed
                    StartCoroutine(importFromFolder(trainingFolderPaths, allData));


                    finished = false;
                }
                else
                {
                    startTime = System.DateTime.UtcNow;
                    StartCoroutine(createSubImages(images, subImageSize, miniImages));
                    finished = false;
                }
            }

           
           
        }
    }

    IEnumerator loadImagesFromFile ()
    {
        Image2[] images = LoadImages();

        
        for (int i = 0; i < images.Length; i ++)
        {
            byte[] bytes = images[i].ConvertToTexture2D().EncodeToPNG();

            File.WriteAllBytes(imageProcessingPath + "/" + images[i].label + "/" + "Image-" + i + ".png", bytes);

            Percent.text = (float)i / images.Length * 100 + " % ";
            PercentSlider.value = (float)i / images.Length;
            yield return null;
        }
        

    }

    public IEnumerator importFromFolder (List<string> paths, List<DataPoint> data )
    {

        createLine("Starting Image Importing");
        List<List<Texture2D>> newImages = new List<List<Texture2D>>();
        List<DataPoint> evalData = new List<DataPoint>();
        int testingIndex = 0;

        for (int i = 0; i < paths.Count; i ++)
        {
            //Add new list
            newImages.Add(new List<Texture2D>());

            //Load images
            newImages[i] = Resources.LoadAll<Texture2D>(paths[i]).ToList();

            testingIndex = Mathf.FloorToInt(newImages[i].Count * trainingSplit);

            //Loop through all images
            for (int j = 0; j < newImages[i].Count; j ++)
            {
                System.Random rng = new System.Random(Random.Range(0, 1000));
                Texture2D img = newImages[i][j];

                //Debug.Log(RandomInNormalDistribution(rng));
                
                if (j >= testingIndex)
                {
                    evalData.Add(imageToData(img, i, outputNum));
                } else
                {
                    //Check if we process the images
                    if (processImages)
                    {
                        //Maybe remove the thresholds
                        //Process images individually
                       // if (rng.NextDouble() >= 0.5)
                       // {
                            double scale = 1 + RandomInNormalDistribution(rng) * 0.1;

                            //Apply Scale (0.5 - 1.5)
                            img = ApplyScale(img, (float)scale);
                        //}

                       // if (rng.NextDouble() >= 0.5)
                       // {
                           
                       // }

                      //  if (rng.NextDouble() >= 0.5)
                      //  {
                            float angle = (float)RandomInNormalDistribution(rng) * 10;

                            //Apply Rotation
                            img = ApplyRotation(img, angle);

                        //  }
                        //Generate offsetNumbers
                        //Used to be 5
                        int offsetX = Mathf.FloorToInt((float)RandomInNormalDistribution(rng) * (img.width / 10));
                            int offsetY = Mathf.FloorToInt((float)RandomInNormalDistribution(rng) * (img.height / 10));

                            //Apply Offset (max 1/3 width and height)
                            img = ApplyOffset(img, offsetX, offsetY);
                       

                        // if (rng.NextDouble() >= 0.5)
                        // {
                        //Apply Noise
                        img = ApplyNoise(img);
                       // }
                    }

                    //Convert to Data point
                    data.Add(imageToData(img, i, outputNum));

                    if (saveImportedTrainingImages)
                    {
                        //Save images

                        //Save Texture as PNG
                        byte[] bytes = img.EncodeToPNG();

                        File.WriteAllBytes(imageProcessingPath + "/" + "Image-" + i + "-" + j + ".png", bytes);

                        yield return null;
                    }
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
        StartCoroutine(splitData(miniImages, trainingSplit));



    }

    public IEnumerator splitData(List<List<Texture2D>> images, float splitRatio)
    {
        //Get the index for the split ratio

        //Convert to data points, add to evaluate data

        createLine("Splitting Data");

        List<DataPoint> evalData = new List<DataPoint>();

        int index = Mathf.FloorToInt(images[0].Count * splitRatio);

        int length = images[0].Count - index;

        for (int i = 0; i < images.Count; i++)
        {
            for (int j = index; j < images[i].Count; j++)
            {
                //Convert to data and add to List
                evalData.Add(imageToData(images[i][j], i, outputNum));
                images[i].RemoveAt(j);
            }

        }

        //Convert to Array 
        evaluateData = evalData.ToArray();
        yield return null;

        createLine("Finished Splitting Data");

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

            System.Random rng = new System.Random(Random.Range(0, 1000));

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

                    if (rng.NextDouble() >= 0.5)
                    {
                        double scale = rng.NextDouble() + 0.5;

                        //Apply Scale (0.5 - 1.5)
                        newImage = ApplyScale(newImage, (float)scale);
                    }

                    if (rng.NextDouble() >= 0.5)
                    {
                        //Generate offsetNumbers
                        int offsetX = Mathf.FloorToInt((float)rng.NextDouble() * (newImage.width / 5));
                        int offsetY = Mathf.FloorToInt((float)rng.NextDouble() * (newImage.height / 5));

                        //Apply Offset (max 1/3 width and height)
                        newImage = ApplyOffset(newImage, offsetX, offsetY);
                    }

                    if (rng.NextDouble() >= 0.5)
                    {
                        float angle = Random.Range(-45, 45);

                        //Apply Rotation
                        newImage = ApplyRotation(newImage, angle);
                    }

                    if (rng.NextDouble() >= 0.5)
                    {
                        //Apply Noise
                        newImage = ApplyNoise(newImage);
                    }

                    createdImages[imgType].Add(newImage);

                    allimgCount++;
                    imgCount++;

                }
                Percent.text = "Image Processing " + imgType + ": " + (float)j / imgs.Count * 100 + " % ";
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

        miniImages = new List<List<Texture2D>>();

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

                Percent.text = "Data Conversion " + type + ": " + (float)i / procImgs[type].Count * 100 + " % ";
                PercentSlider.value = (float)i / procImgs[type].Count;
                yield return null;

                if (saveImageProcessing)
                {
                    //Save Texture as PNG
                    byte[] bytes = procImgs[type][i].EncodeToPNG();

                    File.WriteAllBytes(imageProcessingPath + "/" + "Image-" + type + "-" + i + ".png", bytes);

                    yield return null;
                }
               
            }
            createLine("Finished Type " + type);

            yield return null;
        }

        createLine("Finished Converting to Data");
        yield return null;

        System.DateTime dataEnd = System.DateTime.UtcNow;

        createLine("Time Elapsed Converting to Data: " + (dataEnd - dataStart));
        yield return null;

        processedImages = new List<List<Texture2D>>();

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

        createLine("Network Info: " + "Size:" + sizeToString(networkSettings.networkSize) + "  NumPerBatch: " + networkSettings.dataPerBatch + "  LearnRate: " + networkSettings.initialLearningRate);

        //Create a new Neural Network
        //NeuralNetwork neuro = new NeuralNetwork(networkSettings.neuralNetSize, hiddenActivation, outputActivation, costType);

        NeuralNetwork neuro = new NeuralNetwork(networkSettings.networkSize, Activation.GetActivationFromType(networkSettings.activationType), Activation.GetActivationFromType(networkSettings.outputActivationType), Cost.GetCostFromType(networkSettings.costType));

        //Set Cost function
        neuro.SetCostFunction(Cost.GetCostFromType(Cost.CostType.MeanSquareError));
       

        createLine("Starting Data Shuffle");
        yield return null;

        int total = allData.Count;
        int count = 0;

        System.Random rng = new System.Random();

        List<DataPoint> shuffledData;
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
            }

            createLine("Epoch: " + epoch);

            //Shuffle Batches
            yield return StartCoroutine(ShuffleArray(batches));

            // StartCoroutine(displayCost(false, false, neuro, evaluateData));

           // StartCoroutine(displayCost(true, false, neuro, evaluateData));

            //Teaching
            for (int i = 0; i < batches.Length; i++)
            {
                neuro.Learn(batches[i].data, (1.0 / (1.0 + networkSettings.learnRateDecay * epoch)) * networkSettings.initialLearningRate, networkSettings.regularization, networkSettings.momentum);

                Percent.text = "Teaching: " + (float)i / numOfBatches * 100 + " % ";
                PercentSlider.value = (float)i / numOfBatches;
                yield return null;
            }

            //  StartCoroutine(displayCost(false, true, neuro, evaluateData));

            yield return StartCoroutine(displayCost(true, true, neuro, evaluateData));

            yield return StartCoroutine(EvaluateNetwork(neuro, evaluateData));

        }

        //Once Finished

        createLine("Total Time elapsed: " + (trainEnd - startTime));
        yield return null;

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


        // Debug.Log(image.width);
        //  Debug.Log(image.width);

        Texture2D newImage = new Texture2D(image.width * scaleFactor, image.height * scaleFactor);


        //  Debug.Log(newImage.width);
        //  Debug.Log(newImage.width);

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
                    } else
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
                
               // createLine("All Cost After: " + neuro.Cost(data));
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
                    //double bestCost = bestNetwork.Cost(allTrainData);

                    //double currentCost = network.Cost(allTrainData);

                   // if (currentCost < bestCost)
                   // {
                        //Replace Network witht the new one
                        bestNetwork = network;
                        createLine("New Best Network Made");

                        saveNetwork(bestNetwork, fileName + " (Best)");
                  // }
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

    public IEnumerator ShuffleArray(DataPoint[] data)
    {
        int elementsRemainingToShuffle = data.Length;
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
            (int, double[]) classify = neuro.Classify(data[i].inputs);

            

            //int classify = neuro.Classify(data[i].inputs);
            int label = data[i].label;

           Debug.Log(classify.Item1 + " : " + label);

            if (classify.Item1 == label)
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

    static double RandomInNormalDistribution(System.Random prng, double mean = 0, double standardDeviation = 1)
    {
        double x1 = 1 - prng.NextDouble();
        double x2 = 1 - prng.NextDouble();

        double y1 = System.Math.Sqrt(-2.0 * System.Math.Log(x1)) * System.Math.Cos(2.0 * System.Math.PI * x2);
        return y1 * standardDeviation + mean;
    }

    
    Image2[] LoadImages()
    {
        List<Image2> allImages = new List<Image2>();

        foreach (var file in dataFiles)
        {
            Image2[] images = LoadImages(file.imageFile.bytes, file.labelFile.bytes);
            allImages.AddRange(images);
        }

        return allImages.ToArray();


        Image2[] LoadImages(byte[] imageData, byte[] labelData)
        {
            int numChannels = (greyscale) ? 1 : 3;
            int bytesPerImage = imageSize * imageSize * numChannels;
            int bytesPerLabel = 1;

            int numImages = imageData.Length / bytesPerImage;
            int numLabels = labelData.Length / bytesPerLabel;
            Debug.Assert(numImages == numLabels, $"Number of images doesn't match number of labels ({numImages} / {numLabels})");

            int dataSetSize = System.Math.Min(numImages, numLabels);
            var images = new Image2[dataSetSize];

            // Scale pixel values from [0, 255] to [0, 1]
            double pixelRangeScale = 1 / 255.0;
            double[] allPixelValues = new double[imageData.Length];

            System.Threading.Tasks.Parallel.For(0, imageData.Length, (i) =>
            {
                allPixelValues[i] = imageData[i] * pixelRangeScale;
            });

            // Create images
            System.Threading.Tasks.Parallel.For(0, numImages, (imageIndex) =>
            {
                int byteOffset = imageIndex * bytesPerImage;
                double[] pixelValues = new double[bytesPerImage];
                System.Array.Copy(allPixelValues, byteOffset, pixelValues, 0, bytesPerImage);
                Image2 image = new Image2(imageSize, greyscale, pixelValues, labelData[imageIndex]);
                images[imageIndex] = image;
            });

            return images;
        }


    }
    




}