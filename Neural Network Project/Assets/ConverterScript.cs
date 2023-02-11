using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;

public class ConverterScript : MonoBehaviour
{


    [SerializeField] Image imageHolder;
    [SerializeField] Image View;

    [SerializeField] Texture2D six;
    [SerializeField] Texture2D four;

    // Assets/New Images/Four
    // Assets/New Images/Six

    // Start is called before the first frame update
    void Start()
    {

        Texture2D newImage = new Texture2D(20, 20);


       
        /*
        for (int i = 0; i < 20; i ++)
        {
            for (int j = 0; j < 20; j++)
            {

                newImage.SetPixel(i, j, six.GetPixel(i, j));

               // six.SetPixel(i, j, Color.green);
            }

        }

        newImage.Apply();

        View.sprite = Sprite.Create(newImage, new Rect(0, 0, newImage.width, newImage.height), new Vector2(0, 0));
        */





       // createSubImages(six, new Vector2Int(20, 20), "Assets/New Images/Six/");
       // createSubImages(four, new Vector2Int(20, 20), "Assets/New Images/Four/");

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void createSubImages (Texture2D image, Vector2Int imgSize, string path)
    {
        //Start with 2 loops that determine starting coordinates

        int imgNum = 0;

        for (int startX = 0; startX < image.width; startX = startX + 20)
        {
            for (int startY = 0; startY < image.height; startY = startY + 20)
            {

                Texture2D newImg = new Texture2D(imgSize.x, imgSize.y, TextureFormat.RGB24, false);

                for (int pixelX = 0; pixelX < imgSize.x; pixelX ++)
                {
                    for (int pixelY = 0; pixelY < imgSize.y; pixelY++)
                    {
                        newImg.SetPixel(pixelX, pixelY, image.GetPixel(startX + pixelX, startY + pixelY));
                    }
                }

                //Save Texture as PNG
                byte[] bytes = newImg.EncodeToPNG();
                var dirPath = path;
               
                File.WriteAllBytes(dirPath + "Image" + imgNum + ".png", bytes);

                imgNum++;
            }
        }

        Debug.Log("Complete");





    }


    //What if we did cyclindrical coordinates





}
