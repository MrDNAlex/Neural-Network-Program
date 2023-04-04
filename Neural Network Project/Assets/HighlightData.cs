using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using FlexUI;
using System.IO;
using System.Linq;
using DNANeuralNet;
using UnityEngine.UI;


public class HighlightData : MonoBehaviour
{

    //Structures
    


    public struct ImageHolder
    {
       public Texture2D imgText;
       public string path;


        public void setInfo (string path)
        {
            this.path = path;

            string str = path.Substring(path.LastIndexOf("Resources"));

            str = str.Replace("Resources", "");

            str = str.Remove(0, 1);

            str = str.Replace(".png", "");

            //Debug.Log(str);

            this.imgText = Resources.Load<Texture2D>(str);
        }

    }


    //Get Images from a folder, convert to texture2D? or sprite, display them, probably Texture2D

    //Create a temporary structure that stores the individual images path with it's texture2d?, when we save the image, it will go through the path and delete the image it used 

    //Display the first image 

    //

    List<ImageHolder> unLabledImages = new List<ImageHolder>();
    int index = -1;
    Vector2Int currentHighlightMin;
    Vector2Int currentHighlightMax;
    Texture2D currentImg;

    List<ImageData> labledImages = new List<ImageData>();



    [Header("Imported Images")]
    [SerializeField] ImageHelper.ImportSettings settings;
    [SerializeField] string imagesPath;
   // [SerializeField] Vector2Int imgSizes;

    [Header("Export")]
    [SerializeField] string exportPath;
    [SerializeField] string fileName;


    [Header("UI Stuff")]
    [SerializeField] RectTransform holder;
    [SerializeField] Image ImageDisp;
    [SerializeField] Button NextBTN;
    [SerializeField] Button ClearBTN;
    [SerializeField] Button SaveBTN;
    [SerializeField] Button FinishBTN;
    [SerializeField] Text PercentText;


    int startNum = 0;


    // Start is called before the first frame update
    void Start()
    {
        setUI();

        getImages();

        NextBTN.onClick.AddListener(nextImage);
        ClearBTN.onClick.AddListener(clearImage);
        SaveBTN.onClick.AddListener(saveImage);
        FinishBTN.onClick.AddListener(finishSave);

        if (settings.files.Length > 0)
        {
            labledImages = ImageHelper.LoadImages(settings);
        } else
        {
            labledImages = new List<ImageData>();
        }

        startNum = unLabledImages.Count;
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

   void setUI ()
    {

        Flex Holder = new Flex(holder, 1);

        Flex ImageDisp = new Flex(Holder.getChild(0), 10, Holder);

        Flex Percent = new Flex(Holder.getChild(1), 1, Holder);
        Flex Buttons = new Flex(Holder.getChild(2), 1, Holder);

        Flex Next = new Flex(Buttons.getChild(0), 1, Buttons);
        Flex Clear = new Flex(Buttons.getChild(1), 1, Buttons);
        Flex Save = new Flex(Buttons.getChild(2), 1, Buttons);
        Flex Finish = new Flex(Buttons.getChild(3), 1, Buttons);

        //ImageDisp.setSelfVerticalPadding(0.1f, 1, 0.1f, 1);

        Holder.setSize(new Vector2(Screen.width, Screen.height));

        ImageDisp.setSize(new Vector2(ImageDisp.size.y * (16f / 9f), ImageDisp.size.y));

        LayoutRebuilder.ForceRebuildLayoutImmediate(Holder.UI);
    }

    public void getImages ()
    {

        List<ImageHolder> imageHolders = new List<ImageHolder>();

        if (Directory.Exists(imagesPath))
        {
            DirectoryInfo d = new DirectoryInfo(imagesPath);

            foreach (var file in d.GetFiles("*.png"))
            {
                //Debug.Log(file);

                //Get the path of the imag

                ImageHolder img = new ImageHolder();

                img.setInfo(file.FullName);

                imageHolders.Add(img);
            }
        }

      //  StartCoroutine(ShuffleArray(imageHolders));

        //ImageDisp.sprite = Sprite.Create(imageHolders[0].imgText, new Rect(new Vector2(0, 0), new Vector2(imageHolders[0].imgText.width, imageHolders[0].imgText.height)), new Vector2(0, 0));

        this.unLabledImages = imageHolders;

        nextImage();
    }

    void nextImage ()
    {

        index++;
        currentImg = unLabledImages[index].imgText;
        currentHighlightMax = Vector2Int.zero;
        currentHighlightMin = Vector2Int.zero;

        ImageDisp.sprite = Sprite.Create(currentImg, new Rect(new Vector2(0, 0), new Vector2(currentImg.width, currentImg.height)), new Vector2(0, 0));

       // Destroy(ImageDisp.GetComponent<HighlightProcessor>());

       // ImageDisp.gameObject.AddComponent<HighlightProcessor>();

        ImageDisp.GetComponent<HighlightProcessor>().originalImage = currentImg;

        ImageDisp.GetComponent<HighlightProcessor>().sendScene(this);

        ImageDisp.GetComponent<HighlightProcessor>().getInitSettings();


        PercentText.text = index + "/" + startNum + "    :     " + (float)index / startNum * 100 + "%";

    }

    void clearImage ()
    {
        destroyChildren(ImageDisp.gameObject);
    }

    public void givePos (Vector2Int start, Vector2Int end)
    {
        this.currentHighlightMax = Vector2Int.Max(start, end);
        this.currentHighlightMin = Vector2Int.Min(start, end);

       // Debug.Log("Min: " +currentHighlightMin);
       // Debug.Log("Max: " +currentHighlightMax);
    }

    void saveImage ()
    {

        double[] outputs = new double[4];

        //0 = minx
        //1 = miny
        //2 = maxx
        //3 = maxy

        outputs[0] = currentHighlightMin.x;
        outputs[1] = currentHighlightMin.y;
        outputs[2] = currentHighlightMax.x;
        outputs[3] = currentHighlightMax.y;

        ImageData imageData = new ImageData(unLabledImages[index].imgText, outputs, 0, false, false);

        labledImages.Add(imageData);

        File.Delete(unLabledImages[index].path);

        clearImage();

        nextImage();
    }

    void finishSave ()
    {
        ImageHelper.SaveImages(exportPath, fileName + "_Images", labledImages);

        ImageHelper.SaveLabels(exportPath, fileName + "_Labels", labledImages);

        ImageHelper.SaveOutputs(exportPath, fileName + "_Outputs", labledImages);
    }

    //
    //Extra
    //

    public IEnumerator ShuffleArray(List<ImageHolder> data)
    {
        int elementsRemainingToShuffle = data.Count;
        int randomIndex = 0;
        System.Random prng = new System.Random();

        while (elementsRemainingToShuffle > 1)
        {
            // Choose a random element from array
            randomIndex = prng.Next(0, elementsRemainingToShuffle);
            ImageHolder chosenElement = data[randomIndex];

            // Swap the randomly chosen element with the last unshuffled element in the array
            elementsRemainingToShuffle--;
            data[randomIndex] = data[elementsRemainingToShuffle];
            data[elementsRemainingToShuffle] = chosenElement;

            
            yield return null;
        }
    }

    public void destroyChildren(GameObject Obj)
    {
        //Only deletes children under the one referenced
        foreach (Transform child in Obj.transform)
        {
            //Safe to delete
            Destroy(child.gameObject);
        }
    }






}
