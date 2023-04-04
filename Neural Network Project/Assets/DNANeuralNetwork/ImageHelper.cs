using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Linq;
using System;


namespace DNANeuralNet
{
    public class ImageHelper
    {
        [System.Serializable]
        public struct DataFile
        {
            public TextAsset imageFile;
            public TextAsset labelFile;
            public TextAsset outputFile;
        }

        [System.Serializable]
        public struct ImportSettings
        {
            public DataFile[] files;
            public Vector2Int imageSize;
            public int numOfOutputs;
            public bool greyScale;
            public bool useLabel;
            public bool editImages;

            [Header("SaveMemory")]
            public bool saveMemory;
            public string folderPath;
            //Create a new file in this directory
           
        }

        public static void SaveImages(string path, string name, List<ImageData> images)
        {
            byte[] imageBytes = ImagesToBytes(images.ToArray());

           // Debug.Log("Finished Converting to Bytes");

            SaveBytesToFile(path, name, imageBytes);

           // Debug.Log("Finished Saving");
        }

        public static void SaveLabels(string path, string name, List<ImageData> images)
        {
            byte[] labelBytes = LabelsToBytes(images.ToArray());

           // Debug.Log("Finished Converting to Bytes");

            SaveBytesToFile(path, name, labelBytes);

           // Debug.Log("Finished Saving");
        }

        public static void SaveOutputs(string path, string name, List<ImageData> images)
        {
            byte[] outputBytes = OutputsToBytes(images.ToArray());

           // Debug.Log("Finished Converting to Bytes");

            SaveBytesToFile(path, name, outputBytes);

           // Debug.Log("Finished Saving");
        }

        public static byte[] ImageToBytes(ImageData image)
        {
            byte[] bytes = new byte[image.pixelVals.Length];

            for (int i = 0; i < image.pixelVals.Length; i++)
            {
                bytes[i] = (byte)(image.pixelVals[i] * 255);
            }

            return bytes;
        }

        public static byte[] ImagesToBytes(ImageData[] images)
        {
            List<byte> allImageBytes = new List<byte>();
            foreach (var image in images)
            {
                allImageBytes.AddRange(ImageToBytes(image));
            }
            return allImageBytes.ToArray();
        }

        public static byte[] LabelToBytes(ImageData image)
        {
            List<byte> bytes = new List<byte>();

            //Debug.Log(image.label);

            //Add Label
            bytes.Add((byte)image.label);

            return bytes.ToArray();
        }

        public static byte[] LabelsToBytes(ImageData[] images)
        {
            List<byte> allLabelBytes = new List<byte>();

            foreach (ImageData img in images)
            {
                allLabelBytes.AddRange(LabelToBytes(img));
            }

            return allLabelBytes.ToArray();
        }

        public static byte[] OutputToBytes(ImageData image)
        {
            List<byte> bytes = new List<byte>();

            //Add Outputs
            for (int i = 0; i < image.expectedOutputs.Length; i++)
            {
                byte[] dub = new byte[8];

                dub = BitConverter.GetBytes(image.expectedOutputs[i]);

                bytes.AddRange(dub);
            }

            return bytes.ToArray();
        }

        public static byte[] OutputsToBytes(ImageData[] images)
        {
            List<byte> allOutputBytes = new List<byte>();

            foreach (ImageData img in images)
            {
                allOutputBytes.AddRange(OutputToBytes(img));
            }

            return allOutputBytes.ToArray();
        }

        public static void SaveBytesToFile(string path, string name, byte[] bytes)
        {
            string fullPath = Path.Combine(path, name + ".bytes");

            using (BinaryWriter writer = new BinaryWriter(File.Open(fullPath, FileMode.Create)))
            {
                writer.Write(bytes);
            }
        }

        public static double[] BytesToDouble(int numOfDoubles, byte[] data)
        {
            int bytesPerDouble = 8;

            double[] doubles = new double[numOfDoubles];

            System.Threading.Tasks.Parallel.For(0, numOfDoubles, (i) =>
            {
                int byteOffset = i * bytesPerDouble;

                byte[] bytesForDouble = new byte[bytesPerDouble];

                System.Array.Copy(data, byteOffset, bytesForDouble, 0, bytesPerDouble);

                doubles[i] = BitConverter.ToDouble(bytesForDouble);

            });

            return doubles;
        }

        public static ImageData GetImage (ImportSettings settings, string path)
        {
            byte[] bytes = File.ReadAllBytes(path);

            int numChannels = (settings.greyScale) ? 1 : 3;

            int bytesPerImage = settings.imageSize.x * settings.imageSize.y * numChannels;

            int bytesPerLabel = 1;

            int bytesPerOutput = settings.numOfOutputs * 8;

            double[] pixelVals = new double[bytesPerImage];

            byte[] outputBytes = new byte[bytesPerOutput];
           
            int label = 0;

            //Get Pixel Vals
            double pixelRangeScale = 1 / 255.0;
            for (int i = 0; i < bytesPerImage; i ++)
            {
                pixelVals[i] = bytes[i] * pixelRangeScale;
            }

            //Get output

            for (int i = 0; i < bytesPerOutput; i ++)
            {
                outputBytes[i] = bytes[i + bytesPerImage];
            }

            double[] outputs = BytesToDouble(settings.numOfOutputs, outputBytes);

            label = (int)bytes[bytesPerImage + bytesPerOutput];

            ImageData image = new ImageData(pixelVals, outputs, label, settings.imageSize, false, settings.greyScale);

            //Pixels
            //Output
            //Label

            return image;
        }

        public static void LoadImagesMemory (ImportSettings settings, float trainingSplit)
        {

            string folderPath = settings.folderPath;
            string evalFolderPath = folderPath + "/" + "EvalData";
            string trainFolderPath = folderPath + "/" + "TrainFolder";
            string batchesFolderPath = folderPath + "/" + "Batches";
            
            var mainFolder = Directory.CreateDirectory(folderPath);
            var evalFolder = Directory.CreateDirectory(evalFolderPath);
            var trainFolder = Directory.CreateDirectory(trainFolderPath);
            var batchesFolder = Directory.CreateDirectory(batchesFolderPath);

            int fileIndex = 0;

            foreach (var file in settings.files)
            {
                LoadImages(file.imageFile.bytes, file.labelFile.bytes, file.outputFile.bytes);

            }

            void LoadImages(byte[] imageData, byte[] labelData, byte[] outputData)
            {
                int bytesPerDouble = 8;

                int numChannels = (settings.greyScale) ? 1 : 3;

                //Debug.Log("Channels: " + numChannels);

                int bytesPerImage = settings.imageSize.x * settings.imageSize.y * numChannels;
                //Debug.Log("Bytes: " + bytesPerImage);

                int bytesPerLabel = 1;
                // Mult by 8 for doubles
                int bytesPerOutput = settings.numOfOutputs * 8;

                int numImages = imageData.Length / bytesPerImage;
                int numLabels = labelData.Length / bytesPerLabel;
                int numOutputs = outputData.Length / bytesPerOutput;

              //  Debug.Log("Images: " + numImages);
               // Debug.Log("Labels: " + numLabels);
               // Debug.Log("Outputs: " + numOutputs);


                Debug.Assert(numImages == numLabels, $"Number of images doesn't match number of labels ({numImages} / {numLabels})");

                int splitIndex = Mathf.FloorToInt(numImages * trainingSplit);


                int dataSetSize = System.Math.Min(numImages, numLabels);
                ImageData[] images = new ImageData[dataSetSize];

                //Convert pixel value to double again
                //double pixelRangeScale = 1 / 255.0;

                System.Threading.Tasks.Parallel.For(0, numImages, (imageIndex) =>
                {
                    List<byte> bytes = new List<byte>();

                    int imageByteOffset = imageIndex * bytesPerImage;
                    int outputByteOffset = imageIndex * bytesPerOutput;

                    //Save the new things as bytes

                    //Get pixel vals
                    byte[] pixelVals = new byte[bytesPerImage];
                    for (int i = 0; i < bytesPerImage; i ++)
                    {
                        pixelVals[i] = imageData[i + imageByteOffset];
                    }

                    //Get output
                    byte[] outputBytes = new byte[bytesPerOutput]; 

                    //Get the necessary bytes
                    for (int i = 0; i < bytesPerOutput; i++)
                    {
                        outputBytes[i] = outputData[i + outputByteOffset];
                    }

                    /*
                    double[] outputVals = new double[settings.numOfOutputs];

                    for (int i = 0; i < settings.numOfOutputs; i ++)
                    {
                        int byteOffset = i * bytesPerDouble;

                        byte[] doubleBytes = new byte[bytesPerDouble];

                        System.Array.Copy(outputBytes, byteOffset, doubleBytes, 0, bytesPerDouble);

                        outputVals[i] = BitConverter.ToDouble(doubleBytes);
                    }
                    */

                    //Get label
                    byte[] label = new byte[1];
                    label[0] = labelData[imageIndex];

                    //Add bytes
                    bytes.AddRange(pixelVals);
                    bytes.AddRange(outputBytes);
                    bytes.AddRange(label);

                    var dir = "";

                    if (imageIndex > splitIndex)
                    {
                       // dir = evalFolderPath + "/" + "Image-" + fileIndex + "-" + imageIndex + ".json";
                        SaveBytesToFile(evalFolderPath, "Image-" + fileIndex + "-" + imageIndex, bytes.ToArray());
                    }
                    else
                    {

                        //dir = trainFolderPath + "/" + "Image-" + fileIndex + "-" + imageIndex + ".json";
                        SaveBytesToFile(trainFolderPath, "Image-" + fileIndex + "-" + imageIndex, bytes.ToArray());
                    }

                    //ImageData image = new ImageData(pixelVals, outputVals, label, settings.imageSize, settings.useLabel, settings.greyScale);

                    //Save the data point to it's own file

                    /*
                    var dir = "";

                    if (imageIndex > splitIndex)
                    {
                        dir = evalFolderPath + "/" + "Image-" + fileIndex + "-"+ imageIndex + ".json";

                    } else
                    {
                        
                        dir = trainFolderPath + "/" + "Image-" + fileIndex + "-" + imageIndex + ".json";
                    }
                    string jsonData = JsonUtility.ToJson(image.GetDataPoint(), true);

                    //Debug.Log(jsonData);
                    File.WriteAllText(dir, jsonData);
                    */

                });

                fileIndex++;

                //We are going 480p 

                //852 x 480 


            }

        }

        public static List<ImageData> LoadImages(ImportSettings settings)
        {
            List<ImageData> allImages = new List<ImageData>();

            foreach (var file in settings.files)
            {
                ImageData[] images = LoadImages(file.imageFile.bytes, file.labelFile.bytes, file.outputFile.bytes);
                allImages.AddRange(images);
            }

            return allImages;

            ImageData[] LoadImages(byte[] imageData, byte[] labelData, byte[] outputData)
            {
                int bytesPerDouble = 8;

                int numChannels = (settings.greyScale) ? 1 : 3;

                Debug.Log("Channels: " + numChannels);

                int bytesPerImage = settings.imageSize.x * settings.imageSize.y * numChannels;
                Debug.Log("Bytes: " + bytesPerImage);

                int bytesPerLabel = 1;
                // Mult by 8 for doubles
                int bytesPerOutput = settings.numOfOutputs * 8;

                int numImages = imageData.Length / bytesPerImage;
                int numLabels = labelData.Length / bytesPerLabel;
                int numOutputs = outputData.Length / bytesPerOutput;

                Debug.Log("Images: " + numImages);
                Debug.Log("Labels: " + numLabels);
                Debug.Log("Outputs: " + numOutputs);


                Debug.Assert(numImages == numLabels, $"Number of images doesn't match number of labels ({numImages} / {numLabels})");

                int dataSetSize = System.Math.Min(numImages, numLabels);
                ImageData[] images = new ImageData[dataSetSize];

                //Convert pixel value to double again
                double pixelRangeScale = 1 / 255.0;
                double[] allPixelValues = new double[imageData.Length];

                System.Threading.Tasks.Parallel.For(0, imageData.Length, (i) =>
                {
                    allPixelValues[i] = imageData[i] * pixelRangeScale;
                });

                //Convert bytes to doubles for outputs

                double[] allOutputVals = new double[outputData.Length / bytesPerDouble];

                System.Threading.Tasks.Parallel.For(0, outputData.Length / bytesPerDouble, (i) =>
                {
                    int byteOffset = i * bytesPerDouble;

                    byte[] bytesForDouble = new byte[bytesPerDouble];

                    System.Array.Copy(outputData, byteOffset, bytesForDouble, 0, bytesPerDouble);

                    allOutputVals[i] = BitConverter.ToDouble(bytesForDouble);

                });

                // Create images
                System.Threading.Tasks.Parallel.For(0, numImages, (imageIndex) =>
                {
                    int imageByteOffset = imageIndex * bytesPerImage;
                    int outputByteOffset = imageIndex * settings.numOfOutputs;

                    double[] outputVals = new double[settings.numOfOutputs];
                    System.Array.Copy(allOutputVals, outputByteOffset, outputVals, 0, settings.numOfOutputs);

                    double[] pixelValues = new double[bytesPerImage];
                    System.Array.Copy(allPixelValues, imageByteOffset, pixelValues, 0, bytesPerImage);

                    ImageData image = new ImageData(pixelValues, outputVals, labelData[imageIndex], settings.imageSize, settings.useLabel, settings.greyScale);

                    images[imageIndex] = image;
                });

                return images;

            }
        }

        public static List<ImageData> LoadBatchImages(DataFile file, ImportSettings settings)
        {
            List<ImageData> allImages = new List<ImageData>();


            ImageData[] images = LoadImages(file.imageFile.bytes, file.labelFile.bytes, file.outputFile.bytes);
            allImages.AddRange(images);

            return allImages;

            ImageData[] LoadImages(byte[] imageData, byte[] labelData, byte[] outputData)
            {
                int bytesPerDouble = 8;

                int numChannels = (settings.greyScale) ? 1 : 3;

               // Debug.Log("Channels: " + numChannels);

                int bytesPerImage = settings.imageSize.x * settings.imageSize.y * numChannels;
               // Debug.Log("Bytes: " + bytesPerImage);

                int bytesPerLabel = 1;
                // Mult by 8 for doubles
                int bytesPerOutput = settings.numOfOutputs * 8;

                int numImages = imageData.Length / bytesPerImage;
                int numLabels = labelData.Length / bytesPerLabel;
                int numOutputs = outputData.Length / bytesPerOutput;

               // Debug.Log("Images: " + numImages);
              //  Debug.Log("Labels: " + numLabels);
               // Debug.Log("Outputs: " + numOutputs);

                Debug.Assert(numImages == numLabels, $"Number of images doesn't match number of labels ({numImages} / {numLabels})");

                int dataSetSize = System.Math.Min(numImages, numLabels);
                ImageData[] images = new ImageData[dataSetSize];

                //Convert pixel value to double again
                double pixelRangeScale = 1 / 255.0;
                double[] allPixelValues = new double[imageData.Length];

                System.Threading.Tasks.Parallel.For(0, imageData.Length, (i) =>
                {
                    allPixelValues[i] = imageData[i] * pixelRangeScale;
                });

                //Convert bytes to doubles for outputs

                double[] allOutputVals = new double[outputData.Length / bytesPerDouble];

                System.Threading.Tasks.Parallel.For(0, outputData.Length / bytesPerDouble, (i) =>
                {
                    int byteOffset = i * bytesPerDouble;

                    byte[] bytesForDouble = new byte[bytesPerDouble];

                    System.Array.Copy(outputData, byteOffset, bytesForDouble, 0, bytesPerDouble);

                    allOutputVals[i] = BitConverter.ToDouble(bytesForDouble);

                });

                // Create images
                System.Threading.Tasks.Parallel.For(0, numImages, (imageIndex) =>
                {
                    int imageByteOffset = imageIndex * bytesPerImage;
                    int outputByteOffset = imageIndex * settings.numOfOutputs;

                    double[] outputVals = new double[settings.numOfOutputs];
                    System.Array.Copy(allOutputVals, outputByteOffset, outputVals, 0, settings.numOfOutputs);

                    double[] pixelValues = new double[bytesPerImage];
                    System.Array.Copy(allPixelValues, imageByteOffset, pixelValues, 0, bytesPerImage);

                    ImageData image = new ImageData(pixelValues, outputVals, labelData[imageIndex], settings.imageSize, settings.useLabel, settings.greyScale);

                    images[imageIndex] = image;
                });

                return images;

            }
        }



    }
}

