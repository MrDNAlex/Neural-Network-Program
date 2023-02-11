using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using UnityEditor;
using System.Linq;



public class ImageProcessing : MonoBehaviour
{

    [SerializeField] Texture2D image;
    [SerializeField] Texture2D rotImage;
    [SerializeField] Texture2D fun;
    [SerializeField] float angle;
    [SerializeField] Texture2D expand;
    [SerializeField] Texture2D compress;
   
    //Design a system that gets the center location of the image

    //The for every pixel it grabs the color and position in cartesian

    //then it converts 



    //Use the figured out rotational matrix we found...



    //If All else fails maybe the first paragraph of this will come in handy

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


        createVersions();

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
                // Debug.Log("Pos");
               //  Debug.Log(xIndex);
                // Debug.Log(yIndex);


                int xCenter = Mathf.FloorToInt(image.width/2);
                int yCenter = Mathf.FloorToInt(image.height / 2);

                int translatedX = xIndex - xCenter;
                int translatedY = yIndex - yCenter;


                int oldX = Mathf.FloorToInt(translatedX * Mathf.Cos(angRad) - translatedY*Mathf.Sin(angRad));

                int oldY = Mathf.FloorToInt(translatedX * Mathf.Sin(angRad) + translatedY * Mathf.Cos(angRad));
               // Debug.Log("Old Pos");
              //  Debug.Log(oldX);
              //  Debug.Log(oldY);


                newImage.SetPixel(xIndex, yIndex, getValidPixel(image, oldX, oldY));

            }
           // Debug.Log( (float)xIndex / newImage.width * 100 + " % ");
           
        }

        //Save Texture as PNG
       // byte[] bytes = newImage.EncodeToPNG();


        //File.WriteAllBytes("Assets/New Images/Testing/" + name + ".png", bytes);

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

        float mult = Random.Range(1, 5);

        int num = Mathf.FloorToInt((float)(newImage.width * mult / 5));

        //Debug.Log(num);

        for (int i = 0; i < num; i ++)
        {
            int xPos = Mathf.FloorToInt(Random.Range(0, newImage.width));
            int yPos = Mathf.FloorToInt(Random.Range(0, newImage.height));

            float col = (float)Random.Range(0, 255) / 255;

            newImage.SetPixel(xPos, yPos, new Color(col, col, col));

        }

        //Save Texture as PNG
       // byte[] bytes = newImage.EncodeToPNG();


      //  File.WriteAllBytes("Assets/New Images/Testing/" + name + ".png", bytes);

        return newImage;
    }

    public void createVersions ()
    {
        //Grab all images from the folder

        List<Texture2D> imgs = Resources.LoadAll<Texture2D>("Six").ToList();

        Debug.Log(imgs.Count);

        int imgCount = 0;

        for (int j = 0; j < imgs.Count; j ++)
        {
            //Number of copies
            int num = Random.Range(4, 8);

            for (int i = 0; i < num; i++)
            {
                //Copy the image
                Texture2D newImage = imgs[j];

                //Get a random rotation 

                float angle = Random.Range(0, 360);

                newImage = ApplyRotation(newImage, angle);

                newImage = ApplyNoise(newImage);

                byte[] bytes = newImage.EncodeToPNG();


                File.WriteAllBytes("Assets/Resources/NewSix/" + "Image" + imgCount + ".png", bytes);

                imgCount++;

            }

            Debug.Log((float)j / imgs.Count * 100 + " % ");

        }

    }
    








}
