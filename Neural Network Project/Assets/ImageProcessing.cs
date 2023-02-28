using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using UnityEditor;
using System.Linq;



public class ImageProcessing : MonoBehaviour
{
    //From Resource Folder
    public string importPath;

    //From Assets folder
    public string exportPath;

    public int maxCopies;
    public int minCopies;

    [Range(0, 1)] public double noiseProbability;
    [Range(0, 1)] public double noiseStrength;

    [Header("UI Stuff")]
    [SerializeField] Button StartBTN;
    [SerializeField] Text Log;
    [SerializeField] Text Percent;
    [SerializeField] Slider PercentSlider;
   
    // https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions


    // Start is called before the first frame update
    void Start()
    {

        //Don't forgte to convert from degree to radians

        // float radFactor = Mathf.PI / 180;

        //  Debug.Log(Mathf.Cos(0 * radFactor));
        //  Debug.Log(Mathf.Cos(45 * radFactor));
        //  Debug.Log(Mathf.Cos(90 * radFactor));
        //  Debug.Log(Mathf.Cos(135 * radFactor));
        //  Debug.Log(Mathf.Cos(180 * radFactor));



        // ExpandImage(expand, 5, "expImg");



        // StartCoroutine(ApplyRotation(image, angle, "img"));

        // StartCoroutine(ApplyRotation(fun, angle, "fun"));


        //I Guess we don't need to expand, saves on processing power too



        //Compress(compress, 5, "compRot");

        //applyNoise(compress, "noise");

        //Tomorrow make a order a program so that it takes in the folder with path to all images, then for every image it does between like 3-10 images where it gives random angle and noise, then it saves in the new folders


        //createVersions();

        StartBTN.onClick.AddListener(ProcessImages);

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void ExpandImage (Texture2D image, int scaleFactor, string name)
    {
        //Create a copy image of expanded dimensions


        Debug.Log(image.width);
        Debug.Log(image.width);

        Texture2D newImage = new Texture2D(image.width * scaleFactor, image.height * scaleFactor);


        Debug.Log(newImage.width);
        Debug.Log(newImage.width);

        for (int xIndex = 0; xIndex < newImage.width; xIndex ++)
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

    public Texture2D ApplyRotation (Texture2D image, float angle)
    {
        float angRad = (Mathf.PI/ 180) * angle;

        Texture2D newImage = new Texture2D(image.width, image.height);


        for (int xIndex = 0; xIndex < newImage.width; xIndex++)
        {
            for (int yIndex = 0; yIndex < newImage.height; yIndex++)
            {
               
                int xCenter = Mathf.FloorToInt(image.width/2);
                int yCenter = Mathf.FloorToInt(image.height / 2);

                int translatedX = xIndex - xCenter;
                int translatedY = yIndex - yCenter;


                int oldX = Mathf.FloorToInt(translatedX * Mathf.Cos(angRad) - translatedY*Mathf.Sin(angRad));

                int oldY = Mathf.FloorToInt(translatedX * Mathf.Sin(angRad) + translatedY * Mathf.Cos(angRad));
              
                newImage.SetPixel(xIndex, yIndex, getValidPixel(image, oldX, oldY));

            }
        }
        return newImage;
    }

    public Texture2D scaleImage (Texture2D image, float scaleMult)
    {

       // float angRad = (Mathf.PI / 180) * angle;

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

                //Radius = sqrt(x^2 + y^2)
                //Divide by multiplier

                float oldRadius = Mathf.Sqrt((translatedX * translatedX) + (translatedY * translatedY))/scaleMult;

                //Get angle
                float angle = Mathf.Atan2(translatedY, translatedX);



                int oldX = Mathf.FloorToInt(oldRadius * Mathf.Cos(angle));

                int oldY = Mathf.FloorToInt(oldRadius * Mathf.Sin(angle));

                newImage.SetPixel(xIndex, yIndex, getValidPixel(image, oldX, oldY));

            }
        }
        return newImage;


    }

    public Color getValidPixel (Texture2D image, int oldX, int oldY)
    {
        bool verdict = true;

        int x = oldX + Mathf.FloorToInt(image.width / 2);
        int y = oldY + Mathf.FloorToInt(image.height / 2);

         if ((x < image.width && x >= 0) && (y < image.height && y >= 0))
        {
            //Debug.Log("Hi");
            return image.GetPixel(x, y);
        } else
        {
           // Debug.Log("White");
            return Color.white;
        }

    }

    
    public void Compress (Texture2D image, int scaleFactor, string name)
    {
        Texture2D newImage = new Texture2D(image.width / scaleFactor, image.height / scaleFactor);

        for (int xIndex = 0; xIndex < newImage.width; xIndex++)
        {
            for (int yIndex = 0; yIndex < newImage.height; yIndex++)
            {

                float red = 0;
                float green = 0;
                float blue = 0;

                for (int i = 0; i < scaleFactor; i ++)
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

    public Texture2D ApplyNoise (Texture2D image)
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

        return newImage;
    }

    public IEnumerator createVersions ()
    {
        //Grab all images from the folder

        Log.text = "";

        List<Texture2D> imgs = Resources.LoadAll<Texture2D>(importPath).ToList();

        Log.text += "Images Loaded";
        yield return null;

        int imgCount = 0;

        for (int j = 0; j < imgs.Count; j ++)
        {
            //Number of copies
            int num = Random.Range(minCopies, maxCopies);

            for (int i = 0; i < num; i++)
            {
                //Copy the image
                Texture2D newImage = imgs[j];

                //Get a random rotation 

                float angle = Random.Range(-30, 50);

                newImage = ApplyRotation(newImage, angle);

                newImage = ApplyNoise(newImage);

                byte[] bytes = newImage.EncodeToPNG();


                File.WriteAllBytes(exportPath + "/" + "Image" + imgCount + ".png", bytes);

                imgCount++;

            }

            //Debug.Log((float)j / imgs.Count * 100 + " % ");
            //Log.text += "\n" + (float)j / imgs.Count * 100 + " % ";
            PercentSlider.value = (float)j / imgs.Count;
            yield return null;

        }

        Log.text += "\n" + "Finished";


    }

    public void ProcessImages ()
    {
        StartCoroutine(createVersions());

    }
    








}
