using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Linq;
using UnityEngine.UI;
using DNANeuralNetwork;

public class ClassifyAndTest : MonoBehaviour
{

    public string NeuralNetPath;

    public List<string> testingImagePaths = new List<string>();

    //public string ImagePaths;

    NeuralNetwork neuro;

    NeuralNetworkSaver savedNeuro;

    List<Texture2D> images = new List<Texture2D>();

    List<DataPoint> data = new List<DataPoint>();

    [SerializeField] List<string> dispLabel = new List<string>();

  //  [SerializeField] Dictionary<int, string> labels;


    [SerializeField] Image dispImg;
    [SerializeField] Text guess;
    [SerializeField] Button next;
    [SerializeField] Button load;


    int index = 0;

    bool imagesLoaded = false;
    bool dispFirst = false;


    //Load the neural network with a command

    //Load all the images to try out on

    //Display image and the computers guess

    //Have a button to set for the next guess

    //(Shuffle images)



    // Start is called before the first frame update
    void Start()
    {

        next.onClick.AddListener(nextImage);
        load.onClick.AddListener(LoadStuff);

    }

    public void LoadStuff ()
    {
   
        neuro = loadSaveFromPathNeuralNetwork(NeuralNetPath);

        neuro.SetActivationFunction(Activation.GetActivationFromType(Activation.ActivationType.ReLU), Activation.GetActivationFromType(Activation.ActivationType.Softmax));

        Debug.Log("Neuro Loaded");



        if (imagesLoaded == false)
        {
            //Load Images


            for (int i = 0; i < testingImagePaths.Count; i++)
            {
                //Load Fours first
                List<Texture2D> loadedImages = Resources.LoadAll<Texture2D>(testingImagePaths[i]).ToList();

                foreach (Texture2D img in loadedImages)
                {
                    //Convert to Datapoint
                    data.Add(imageToData(img, i, testingImagePaths.Count));
                }
            }


            for (int i = 0; i < testingImagePaths.Count; i++)
            {
                //Load Fours first
                List<Texture2D> loadedImages = Resources.LoadAll<Texture2D>(testingImagePaths[i]).ToList();

                foreach (Texture2D img in loadedImages)
                {
                    images.Add(img);
                }
            }


            Debug.Log(images.Count);
            Debug.Log(data.Count);

            imagesLoaded = true;
        }


        List<DataPoint> shuffle = new List<DataPoint>();
        List<Texture2D> imgs = new List<Texture2D>();


        while (data.Count > 0)
        {
            int ran = Random.Range(0, data.Count);

            shuffle.Add(data[ran]);
            imgs.Add(images[ran]);

            data.RemoveAt(ran);
            images.RemoveAt(ran);

        }

        data = shuffle;
        images = imgs;

        Debug.Log("Shuffle Done");

        nextImage();

    }

    // Update is called once per frame
    void Update()
    {
        
        if (Input.GetKey(KeyCode.LeftControl) && Input.GetKey(KeyCode.L))
        {
            //Load Neural Network

            savedNeuro = loadSaveFromPath(NeuralNetPath);

           // neuro = savedNeuro.createNetwork(savedNeuro);

            Debug.Log("Neuro Loaded");
        }

        if (Input.GetKey(KeyCode.LeftControl) && Input.GetKey(KeyCode.I))
        {
            if (imagesLoaded == false)
            {
                //Load Images


                for (int i = 0; i < testingImagePaths.Count; i++)
                {
                    //Load Fours first
                    List<Texture2D> loadedImages = Resources.LoadAll<Texture2D>(testingImagePaths[i]).ToList();

                    foreach (Texture2D img in loadedImages)
                    {
                        //Convert to Datapoint
                        data.Add(imageToData(img, i, testingImagePaths.Count));
                    }
                }


                for (int i = 0; i < testingImagePaths.Count; i++)
                {
                    //Load Fours first
                    List<Texture2D> loadedImages = Resources.LoadAll<Texture2D>(testingImagePaths[i]).ToList();

                    foreach (Texture2D img in loadedImages)
                    {
                        images.Add(img);
                    }
                }


                Debug.Log(images.Count);
                Debug.Log(data.Count);

                imagesLoaded = true;
            }

        }


        if (Input.GetKey(KeyCode.LeftControl) && Input.GetKey(KeyCode.H))
        {
            //Shuffle data

            List<DataPoint> shuffle = new List<DataPoint>();
            List<Texture2D> imgs = new List<Texture2D>();


            while (data.Count > 0)
            {
                int ran = Random.Range(0, data.Count);

                shuffle.Add(data[ran]);
                imgs.Add(images[ran]);

                data.RemoveAt(ran);
                images.RemoveAt(ran);

            }

            data = shuffle;
            images = imgs;

            Debug.Log("Shuffle Done");
            
        }

        if (Input.GetKey(KeyCode.LeftControl) && Input.GetKey(KeyCode.D))
        {
            if (dispFirst == false)
            {
                nextImage();
                dispFirst = true;
            }
            //Display first image

        }

        if (Input.GetKey(KeyCode.LeftControl) && Input.GetKey(KeyCode.Q))
        {
            DataPointFile file = new DataPointFile();

            foreach (DataPoint d in data)
            {
                file.AddData(d);
            }

            var dir = "Assets/Resources/" + "Data" + ".json";

            string jsonData = JsonUtility.ToJson(file, true);

            Debug.Log(jsonData);

            File.WriteAllText(dir, jsonData);

           
        }

    }

    public NeuralNetworkSaver loadSaveFromPath(string path)
    {
        //This function loads the save named into the currently used save file

        //Debug.Log(path);
        string jsonData = "";
        if (File.Exists(path))
        {
            //Extract JSON Data
            jsonData = File.ReadAllText(path);
            Debug.Log(jsonData);
            return JsonUtility.FromJson<NeuralNetworkSaver>(jsonData);
        }
        else
        {
            Debug.Log("Doesn't exist");
            return null;
        }

    }

    public NeuralNetwork loadSaveFromPathNeuralNetwork(string path)
    {
        //This function loads the save named into the currently used save file

        //Debug.Log(path);
        string jsonData = "";
        if (File.Exists(path))
        {
            //Extract JSON Data
            jsonData = File.ReadAllText(path);
            Debug.Log(jsonData);
            return JsonUtility.FromJson<NeuralNetwork>(jsonData);
        }
        else
        {
            Debug.Log("Doesn't exist");
            return null;
        }

    }


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

    public void nextImage ()
    {

        index++;

        DataPoint imgData = data[index];

        Texture2D img = images[index];


        dispImg.sprite = Sprite.Create(img, new Rect(new Vector2(0,0), new Vector2(img.width, img.height)), new Vector2(0,0));


        (int label, double[] results) = neuro.Classify(imgData.inputs);
      // int result = neuro.Classify(imgData.inputs);

      //  double[] result2 = neuro.Classify2(imgData.inputs);



        guess.text = "DNA-PC guesses this is a " + dispLabel[label];

        guess.text += "\n Answer is a " + dispLabel[imgData.label];


        for (int i = 0; i < results.Length; i ++)
        {
            guess.text += "\n" + dispLabel[i] + " : " + results[i] * 100 + "%";
        }


       // guess.text += "\n" + dispLabel[0] + ": " + result2[0] * 100 + " % ";
       // guess.text += "\n" + dispLabel[1] + ": " + result2[1] * 100 + " % ";
       // guess.text += "\n" + dispLabel[2] + ": " + result2[2] * 100 + " % ";

        



        
        /*
        if (result == 0)
        {
            guess.text = "DNA-PC guesses this is a Four";

            
            //Four
           
        } else
        {
            guess.text = "DNA-PC guesses this is a Six";
            
           
        }

        if (imgData.label == 0)
        {
            guess.text += "\n Answer is a Four";
        }
        else
        {
            guess.text += "\n Answer is a Six";
        }
        */
        

        // Debug.Log(imgData.label);


    }









}
