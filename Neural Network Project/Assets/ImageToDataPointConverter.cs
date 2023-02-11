using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using UnityEditor;
using System.Linq;


public class ImageToDataPointConverter : MonoBehaviour
{
    // Start is called before the first frame update

    //First import all the textures

    //Loop through all of them while also making an array with the normalized pixel values

    public List<DataPoint> allData = new List<DataPoint>();

    public List<Texture2D> allImages = new List<Texture2D>();


    void Start()
    {

        //0 = 4
        //1 = 6

        StartCoroutine(createData());


    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public DataPoint imageToData (Texture2D image, int labelIndex, int labelNum)
    {

        double[] pixels = new double[image.width * image.height];

        for (int x = 0; x < image.width; x ++)
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

    public IEnumerator createData ()
    {
        //Load images
        allImages = Resources.LoadAll<Texture2D>("NewFour").ToList();

        Debug.Log(allImages.Count);

        yield return null;

        //Fours  (0)
        for (int i = 0; i < allImages.Count; i ++)
        {
            allData.Add(imageToData(allImages[i], 0, 2));
            Debug.Log((float)i / allImages.Count * 100 + " % ");
            yield return null;
        }

        //Sixes (1)
        allImages = Resources.LoadAll<Texture2D>("NewSix").ToList();

        Debug.Log(allImages.Count);

        for (int i = 0; i < allImages.Count; i++)
        {
            allData.Add(imageToData(allImages[i], 1, 2));
            Debug.Log((float)i / allImages.Count * 100 + " % ");
            yield return null;
        }

        Debug.Log("Finished");

        DataStorage data = new DataStorage(allData);


        var dir = "Assets/Resources/" + "AllData" + ".json";

        string jsonData = JsonUtility.ToJson(data, true);

        Debug.Log(jsonData);

        File.WriteAllText(dir, jsonData);

        Debug.Log(allData.Count);

    }

}

[System.Serializable]
public class DataStorage
{

    public List<DataPoint> allData = new List<DataPoint>();

    public DataStorage (List<DataPoint> list)
    {
        allData = list;
    }


}




