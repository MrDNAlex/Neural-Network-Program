using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;
using DNANeuralNetwork;
using System.IO;

using UnityEngine.Rendering;


public class ImageToDataFile : MonoBehaviour
{
    [Header("Conversion")]
    [SerializeField] string exportPath;
    [SerializeField] string fileName;
    [SerializeField] List<string> trainingFolderPaths = new List<string>();
    [Range(0, 1)] public float trainingSplit;
    [SerializeField] bool useIndexAsLabel;
    [SerializeField] bool greyScale;

    [Header("Custom Data")]
    [SerializeField] int outputCount;

    [Header("UI Stuff")]
    [SerializeField] Button StartBTN;
    //[SerializeField] Text Log;
    [SerializeField] Text Percent;
    [SerializeField] Slider PercentSlider;
    [SerializeField] GameObject logLine;
    [SerializeField] Transform content;

    MessageLog messageLog = new MessageLog();

    public ImageFiles file;

    // Start is called before the first frame update
    void Start()
    {
        StartBTN.onClick.AddListener(convertToFile);
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void convertToFile()
    {
        StartCoroutine(importFromFolder(trainingFolderPaths));
    }

    public IEnumerator importFromFolder(List<string> paths)
    {

        List<ImageData> imageData = new List<ImageData>();

        file = new ImageFiles();

        createLine("Starting Image Importing");
        List<List<Texture2D>> newImages = new List<List<Texture2D>>();
        //List<DataPoint> evalData = new List<DataPoint>();
        int testingIndex = 0;


        for (int i = 0; i < paths.Count; i++)
        {
            //Add new list
            newImages.Add(new List<Texture2D>());

            //Load images
            newImages[i] = Resources.LoadAll<Texture2D>(paths[i]).ToList();

            //Loop through all images
            for (int j = 0; j < newImages[i].Count; j++)
            {

                //Check for eval Index
                imageData.Add(new ImageData(newImages[i][j], singleOutput(i), i, useIndexAsLabel, greyScale));


                Percent.text = (float)j / newImages[i].Count * 100 + " % ";
                PercentSlider.value = (float)j / newImages[i].Count;
                yield return null;

            }

            createLine("Finished Importing " + i);
            yield return null;
        }

        //Save File at Export path

        ImageHelper.SaveImages(exportPath, fileName + "_Images", imageData);

        ImageHelper.SaveLabels(exportPath, fileName + "_Labels", imageData);

        ImageHelper.SaveOutputs(exportPath, fileName + "_Outputs", imageData);

        createLine("Saved");
    }

    public void createLine(string message)
    {
        GameObject line = Instantiate(logLine, content);

        line.transform.GetChild(0).GetComponent<Text>().text = message;

        messageLog.addMessage(message);

    }

    double[] singleOutput(int index, double value = 1)
    {
        if (useIndexAsLabel)
        {
            //Use Label Index
            double[] output = new double[trainingFolderPaths.Count];

            for (int i = 0; i < output.Length; i ++)
            {
                if (index == i)
                {
                    output[index] = value;
                } else
                {
                    output[index] = 0;
                }
            }

            return output;
        }
        else
        {
            //Don't use label index
            double[] output = new double[outputCount];

            for (int i = 0; i < output.Length; i++)
            {
                if (index == i)
                {
                    output[index] = value;
                }
                else
                {
                    output[index] = 0;
                }
            }

            return output;
        }

    }

    void save(ImageFiles file)
    {
        //Things are too big, so let's create 2 files, the label information and the pixel value information, also convert pixel value from double to int 

        // file.createSave();

        var dir = exportPath + "/" + fileName + ".json";

        string jsonData = JsonUtility.ToJson(file, true);

        Debug.Log(jsonData);

        File.WriteAllText(dir, jsonData);
    }






    //Seb Lague
    /*
     public void ProcessAndSaveAll()
	{
		List<byte> allBytes = new List<byte>();
		List<byte> allLabelBytes = new List<byte>();
		System.Random rng = new System.Random();


		for (int i = 0; i < loader.NumImages; i++)
		{
			for (int j = 0; j < numVersionsPerImg; j++)
			{
				Image image = loader.GetImage(i);
				var settings = CreateRandomSettings(rng, image);
				var transformedImage = TransformImage(image, settings, currentImage.size);
				byte[] bytes = ImageHelper.ImageToBytes(transformedImage);
				allBytes.AddRange(bytes);
				allLabelBytes.Add((byte)image.label);
			}
		}

		FileHelper.SaveBytesToFile(FileHelper.MakePath("Assets"), saveFileName, allBytes.ToArray(), true);
		FileHelper.SaveBytesToFile(FileHelper.MakePath("Assets"), saveFileName + "_labels", allLabelBytes.ToArray(), true);
	}

     */






}
