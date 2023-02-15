using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
public class TotalConversion : MonoBehaviour
{

    //Start With converting big images to smaller ones

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
    //List<Texture2D>  = new List<Texture2D>();
    List<DataPoint> allData = new List<DataPoint>();
    [SerializeField] int outputNum;





    [Header("Train Network")]
    DataPoint[] allTrainData;
    NeuralNetwork neuro;
    DataPoint[,] feedingData;

    [SerializeField] int[] neuralNetworkSize;

    [SerializeField] int numOfBatches = 100;

    [SerializeField] float learnRate;

    //From Asset Folder
    [SerializeField] string exportPath;

    [SerializeField] string fileName;

   




    [Header("UI Stuff")]
    [SerializeField] Button StartBTN;
    [SerializeField] Text Log;
    [SerializeField] Text Percent;
    [SerializeField] Slider PercentSlider;


    


    int step = 0;




    // Start is called before the first frame update
    void Start()
    {
        

        StartBTN.onClick.AddListener(nextStep);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void nextStep ()
    {
        switch (step)
        {
            case 0:
                StartCoroutine(createSubImages(images, subImageSize)); 
                break;
            case 1:
                StartCoroutine(createVersions());
                
                break;
            case 2:
                StartCoroutine(createData(processedImages, allData));
                
                break;
            case 3:
                StartCoroutine(trainNetwork());
                break;
        }

        step++;
    }




    //
    //Convert to Mini
    //

    public IEnumerator createSubImages(List<Texture2D> image, Vector2Int imgSize)
    {
        //Start with 2 loops that determine starting coordinates

        for (int i = 0; i < image.Count; i ++ )
        {

            miniImages.Add(new List<Texture2D>());

            int imgNum = 0;

            int totalNum = (image[i].width / imgSize.x) * (image[i].height / imgSize.y);
           

            for (int startX = 0; startX < image[i].width; startX = startX + 20)
            {
                for (int startY = 0; startY < image[i].height; startY = startY + 20)
                {

                    Texture2D newImg = new Texture2D(imgSize.x, imgSize.y, TextureFormat.RGB24, false);

                    for (int pixelX = 0; pixelX < imgSize.x; pixelX++)
                    {
                        for (int pixelY = 0; pixelY < imgSize.y; pixelY++)
                        {
                            newImg.SetPixel(pixelX, pixelY, image[i].GetPixel(startX + pixelX, startY + pixelY));
                        }
                    }


                    miniImages[i].Add(newImg);

                    imgNum++;
                    Percent.text = (float)imgNum / totalNum * 100 + " % ";
                    PercentSlider.value = (float)imgNum / totalNum;
                    yield return null;

                }
            }

            Log.text += "\n Finished Image " + i ;
            yield return null;

        }


        Log.text += "\n Finished Sub Image Creation";

    }


    //
    //Image Processing
    //

    public IEnumerator createVersions()
    {
        //Grab all images from the folder

        for (int imgType = 0; imgType < miniImages.Count; imgType ++)
        {

            processedImages.Add(new List<Texture2D>());

            List<Texture2D> imgs = miniImages[imgType];

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


                    processedImages[imgType].Add(newImage);


                    imgCount++;

                }

                Percent.text = "\n" + (float)j / imgs.Count * 100 + " % ";
                PercentSlider.value = (float)j / imgs.Count;
                yield return null;

            }

            Log.text += "\n Finished Layer " + imgType;
            yield return null;

        }

        Log.text += "\n" + "Finished Image Processing";


    }

    //
    //Convert To Data
    //

    public IEnumerator createData(List<List<Texture2D>> procImgs, List<DataPoint> allData)
    {
        
        for (int type = 0; type < procImgs.Count; type++)
        {
           
            for (int i = 0; i < procImgs[type].Count; i++)
            {
                allData.Add(imageToData(procImgs[type][i], type, outputNum));

                Percent.text = (float)i / procImgs[type].Count * 100 + " % ";
                PercentSlider.value = (float)i / procImgs[type].Count;
                yield return null;
            }
            Log.text += "\n Finished Type " + type;
            yield return null;
        }


        Log.text += "\n Finished Converting to Data";
        yield return null;

    }


    //
    //Train Network
    //

    public IEnumerator trainNetwork()
    {
       
        int total = allData.Count;
        int count = 0;

        List<DataPoint> shuffledData = new List<DataPoint>();

        //Shuffle Data
        while (allData.Count > 0)
        {
            int ranIndex = Random.Range(0, allData.Count - 1);

            shuffledData.Add(allData[ranIndex]);

            allData.RemoveAt(ranIndex);

            count++;

            Percent.text = (float)count / total * 100 + " % ";
            PercentSlider.value = (float)count / total;

        }

        allData = shuffledData;

        allTrainData = shuffledData.ToArray();

        //Create a new Neural Network
        neuro = new NeuralNetwork(neuralNetworkSize);

        //Calculate cost at first

        int numPerBatch = Mathf.FloorToInt(allTrainData.Length / numOfBatches);

        feedingData = new DataPoint[numOfBatches, numPerBatch];

       

        Debug.Log("Batching");

        for (int i = 0; i < numOfBatches; i++)
        {
            for (int j = 0; j < numPerBatch; j++)
            {
                feedingData[i, j] = allTrainData[numPerBatch * i + j];

            }

        }

        Log.text += "\n Finished Batching";
        yield return null;

        Log.text += "\n Single: " + neuro.Cost(allTrainData[0]);
        yield return null;

        Log.text += "\n All: " + neuro.Cost(allTrainData);
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

        Log.text += "\n Single: " + neuro.Cost(allTrainData[0]);
        yield return null;

        Log.text += "\n All: " + neuro.Cost(allTrainData);
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


        List<DataPoint> testing = new List<DataPoint>();

        for (int i = 0; i < processedImages.Count; i ++)
        {
            for (int j = 0; j < 50; j ++)
            {

                testing.Add(imageToData(processedImages[i][j], i, neuralNetworkSize[neuralNetworkSize.Length - 1]));

            }
        }


       foreach (DataPoint data in testing)
        {
            Debug.Log(neuro.Classify(data.inputs) + " : " + data.label);
        }


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


}
