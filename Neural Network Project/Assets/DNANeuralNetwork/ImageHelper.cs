using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Linq;
using System;


namespace DNANeuralNetwork
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

        }

        public static void SaveImages(string path, string name, List<ImageData> images)
        {
            byte[] imageBytes = ImagesToBytes(images.ToArray());

            Debug.Log("Finished Converting to Bytes");

            SaveBytesToFile(path, name, imageBytes);

            Debug.Log("Finished Saving");
        }

        public static void SaveLabels(string path, string name, List<ImageData> images)
        {
            byte[] labelBytes = LabelsToBytes(images.ToArray());

            Debug.Log("Finished Converting to Bytes");

            SaveBytesToFile(path, name, labelBytes);

            Debug.Log("Finished Saving");
        }

        public static void SaveOutputs(string path, string name, List<ImageData> images)
        {
            byte[] outputBytes = OutputsToBytes(images.ToArray());

            Debug.Log("Finished Converting to Bytes");

            SaveBytesToFile(path, name, outputBytes);

            Debug.Log("Finished Saving");
        }

        public static byte[] ImageToBytes(ImageData image)
        {
            byte[] bytes = new byte[image.numPixels];

            for (int i = 0; i < image.numPixels; i++)
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
                int bytesPerImage = settings.imageSize.x * settings.imageSize.y * numChannels;

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

                    ImageData image = new ImageData(pixelValues, outputVals, (int)labelData[imageIndex], settings.imageSize, settings.useLabel, settings.greyScale);

                    images[imageIndex] = image;
                });

                return images;

            }
        }



    }
}

