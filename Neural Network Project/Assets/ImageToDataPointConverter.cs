using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using UnityEditor;
using System.Linq;


public class ImageToDataPointConverter : MonoBehaviour
{

    //From Resources Folder
    public List<string> importPaths = new List<string>();

   

    public string fileName;

    //From Asset Folder
    public string dataExportPath;


    [Header("UI Stuff")]
    [SerializeField] Button StartBTN;
    [SerializeField] Text Log;
    [SerializeField] Text Percent;
    [SerializeField] Slider PercentSlider;

    List<DataPoint> allData = new List<DataPoint>();

     List<Texture2D> allImages = new List<Texture2D>();


    void Start()
    {

        //0 = 4
        //1 = 6

       // StartCoroutine(createData());

        StartBTN.onClick.AddListener(startConversion);
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
        Log.text = "";
        for (int type = 0; type < importPaths.Count; type ++)
        {
            Log.text += "\n Loading Images from " + importPaths[type];

            yield return null;

            //Load images
            allImages = Resources.LoadAll<Texture2D>(importPaths[type]).ToList();

            Log.text += "\n Images Loaded";

            yield return null;

            for (int i = 0; i < allImages.Count; i++)
            {
                allData.Add(imageToData(allImages[i], type, 2));

                Percent.text = (float)i / allImages.Count * 100 + " % ";
                PercentSlider.value = (float)i / allImages.Count;
                yield return null;
            }

            Log.text += "\n Finished Converting " + importPaths[type];
            yield return null;
        }

       
        Log.text += "\n Finished Processing";
        yield return null;



        DataStorage data = new DataStorage(allData);

        Log.text += "\n Finished Data Storage Conversion";
        yield return null;

        var dir = dataExportPath + "/" + fileName + ".json";

        string jsonData = JsonUtility.ToJson(data, true);

        Debug.Log(jsonData);

        File.WriteAllText(dir, jsonData);

        Log.text += "\n Saved and Finished";
        yield return null;

    }

    public void startConversion ()
    {
        StartCoroutine(createData());
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




