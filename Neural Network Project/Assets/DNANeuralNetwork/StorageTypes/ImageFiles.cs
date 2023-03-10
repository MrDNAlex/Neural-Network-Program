using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Linq;
using System.IO;


[System.Serializable]
public class ImageFiles
{
   

    public List<ImageData> testImages = new List<ImageData>();
    
    public ImageFiles()
    {
        testImages = new List<ImageData>();
       
    }

    public void addTestImage(ImageData data)
    {
        this.testImages.Add(data);
    }

    //Probably move all the Image Saving shit to a new custom saver class

    //Tomorrow, build a scene that can load the info, or implement in the current training stuff


   


    /*
    Image2[] LoadImages(DataFile[] dataFiles)
    {
        List<Image2> allImages = new List<Image2>();

        foreach (var file in dataFiles)
        {
            Image2[] images = LoadImages(file.imageFile.bytes, file.labelFile.bytes);
            allImages.AddRange(images);
        }

        return allImages.ToArray();


        Image2[] LoadImages(byte[] imageData, byte[] labelData)
        {
            int numChannels = (greyscale) ? 1 : 3;
            int bytesPerImage = imageSize * imageSize * numChannels;
            int bytesPerLabel = 1;

            int numImages = imageData.Length / bytesPerImage;
            int numLabels = labelData.Length / bytesPerLabel;
            Debug.Assert(numImages == numLabels, $"Number of images doesn't match number of labels ({numImages} / {numLabels})");

            int dataSetSize = System.Math.Min(numImages, numLabels);
            var images = new Image2[dataSetSize];

            // Scale pixel values from [0, 255] to [0, 1]
            double pixelRangeScale = 1 / 255.0;
            double[] allPixelValues = new double[imageData.Length];

            System.Threading.Tasks.Parallel.For(0, imageData.Length, (i) =>
            {
                allPixelValues[i] = imageData[i] * pixelRangeScale;
            });

            // Create images
            System.Threading.Tasks.Parallel.For(0, numImages, (imageIndex) =>
            {
                int byteOffset = imageIndex * bytesPerImage;
                double[] pixelValues = new double[bytesPerImage];
                System.Array.Copy(allPixelValues, byteOffset, pixelValues, 0, bytesPerImage);
                Image2 image = new Image2(imageSize, greyscale, pixelValues, labelData[imageIndex]);
                images[imageIndex] = image;
            });

            return images;
        }


    }
    */

}


