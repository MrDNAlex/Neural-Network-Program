using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;

public class ConverterScript : MonoBehaviour
{
    //Path from asset folder
    public string exportPath;

    public Vector2Int subImageSize;

   
    [Header("Image")]
    [SerializeField] Texture2D Image;

    [Header("UI Stuff")]
    [SerializeField] Button StartBTN;
    [SerializeField] Text Log;
    [SerializeField] Text Percent;
    [SerializeField] Slider PercentSlider;


    // Assets/New Images/Four
    // Assets/New Images/Six

    // Start is called before the first frame update
    void Start()
    {

       // Texture2D newImage = new Texture2D(20, 20);

       

        StartBTN.onClick.AddListener(Convert);

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public IEnumerator createSubImages (Texture2D image, Vector2Int imgSize, string path)
    {
        //Start with 2 loops that determine starting coordinates

        int imgNum = 0;

        int totalNum = (image.width/imgSize.x) * (image.height/imgSize.y);
        Log.text = "";
        Log.text += "Start";

        for (int startX = 0; startX < image.width; startX = startX + 20)
        {
            for (int startY = 0; startY < image.height; startY = startY + 20)
            {
                bool saveImg = false;

                Texture2D newImg = new Texture2D(imgSize.x, imgSize.y, TextureFormat.RGB24, false);

                for (int pixelX = 0; pixelX < imgSize.x; pixelX ++)
                {
                    for (int pixelY = 0; pixelY < imgSize.y; pixelY++)
                    {
                        newImg.SetPixel(pixelX, pixelY, image.GetPixel(startX + pixelX, startY + pixelY));

                        if (image.GetPixel(startX + pixelX, startY + pixelY).r <= 0.3)
                        {
                            saveImg = true;
                        }
                    }
                }

                //Save Texture as PNG

                if (saveImg)
                {
                    byte[] bytes = newImg.EncodeToPNG();
                    var dirPath = path;

                    File.WriteAllBytes(dirPath + "Image" + imgNum + ".png", bytes);
                }
               

                imgNum++;


                Percent.text = (float)imgNum / totalNum*100 + " % ";
                PercentSlider.value = (float)imgNum / totalNum;
                yield return null;

            }
        }


        Log.text += "\n Finished";
       
    }


    //What if we did cyclindrical coordinates

    public void Convert ()
    {
        StartCoroutine(createSubImages(Image, subImageSize, exportPath + "/"));
    }

}
