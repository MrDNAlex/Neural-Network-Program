using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DNAMath;
using DNANeuralNet;

public class MatrixTesting : MonoBehaviour
{
    [SerializeField] Vector2Int Mat1Dim;

    [SerializeField] Vector2Int Mat2Dim;

    [SerializeField] int[] layerSizes;

    [SerializeField] ComputeShader shader;

    [SerializeField] Texture2D image;

   // public DNANeuralNetworkInfo network;

    System.DateTime startTime;
    System.DateTime endTime;

    // Start is called before the first frame update
    void Start()
    {

        DNANeuralNetwork neuro = new DNANeuralNetwork(layerSizes);

        DNAMatrix inputImg = new DNAMatrix(image.height * image.width, 1);

        for (int i = 0; i < image.width; i++)
        {
            for (int j = 0; j < image.height; j++)
            {
                inputImg[j, i] = image.GetPixel(i, j).r;
            }
        }

        (int index, DNAMatrix output) = neuro.Classify(inputImg);

        output.DisplayMat();

        /*

        DNAMatrix mat1 = new DNAMatrix(Mat1Dim.x, Mat1Dim.y);

        DNAMatrix mat2 = new DNAMatrix(Mat2Dim.x, Mat2Dim.y);

      startTime = System.DateTime.UtcNow;

        //DNAMatrix newMat = mat1 * mat2;

        endTime = System.DateTime.UtcNow;

        Debug.Log("Total Time elapsed CPU (MilliSeconds): " + (endTime - startTime).TotalMilliseconds);

        startTime = System.DateTime.UtcNow;

        DNAMatrix mat = mat1.multMatrixGPU(mat1, mat2);

        endTime = System.DateTime.UtcNow;

        string output = " --> (" + mat.Height + " x " + mat.Width + ")";

        Debug.Log("Total Time elapsed GPU (Setup + Matrix Multiplication) (" + mat1.Height + "x" + mat1.Width + "*" + mat2.Height + "x" + mat2.Width + ") "+ output + ": " + (endTime - startTime).TotalMilliseconds + "(MilliSeconds)");
        
        */
        /*
        DNANeuralNetwork neuro = new DNANeuralNetwork(network);

        Debug.Log(neuro.layers[0].iLayer.getOutputSize(new Vector2Int(28, 28)));

        Vector2Int size = neuro.layers[0].iLayer.getOutputSize(new Vector2Int(28, 28));

        Debug.Log(neuro.layers[1].iLayer.getOutputSize(size));

        size = neuro.layers[1].iLayer.getOutputSize(size);

        Debug.Log(neuro.layers[2].iLayer.getOutputSize(size));
        size = neuro.layers[2].iLayer.getOutputSize(size);

        Debug.Log(neuro.layers[3].iLayer.getOutputSize(size));
        size = neuro.layers[3].iLayer.getOutputSize(size);

        Debug.Log(neuro.layers[4].iLayer.getOutputSize(size));
        size = neuro.layers[4].iLayer.getOutputSize(size);

        DNAMatrix matrix = new DNAMatrix(new Vector2Int(image.width, image.height));

        for (int i = 0; i < image.width; i ++)
        {
            for (int j = 0; j < image.height; j++)
            {
                matrix.setValue(i, j, image.GetPixel(i, j).r);
            }
        }

        for (int i = 0; i < matrix.values.Length; i ++)
        {
            if (matrix.values[i] < 0.1)
            {
                matrix.values[i] = 0;
            } else
            {
                matrix.values[i] = 1;
            }
        }

        Debug.Log(DNAMatrix.displayMat(matrix));

        DNAMatrix[] outputs = neuro.CalculateOutputs(matrix);

        foreach (DNAMatrix mat in outputs)
        {
            Debug.Log(DNAMatrix.displayMat(mat));
        }

        // Debug.Log(neuro.layers[2].iLayer.getOutputSize(size));
        //float[] mat1 = new float[16];

        // float[] mat2 = new float[24];
        */

        /*
        mat1[0] = 1;
        mat1[1] = 2;
        mat1[2] = 3;
        mat1[3] = 4;
        mat1[4] = 5;
        mat1[5] = 6;
        mat1[6] = 7;
        mat1[7] = 8;
        mat1[8] = 9;
        mat1[9] = 10;
        mat1[10] = 11;
        mat1[11] = 12;
        mat1[12] = 13;
        mat1[13] = 14;
        mat1[14] = 15;
        mat1[15] = 16;


        mat2[0] = 1;
        mat2[1] = 2;
        mat2[2] = 3;
        mat2[3] = 4;
        mat2[4] = 5;
        mat2[5] = 6;
        mat2[6] = 7;
        mat2[7] = 8;
        mat2[8] = 9;
        mat2[9] = 10;
        mat2[10] = 11;
        mat2[11] = 12;
        mat2[12] = 13;
        mat2[13] = 14;
        mat2[14] = 15;
        mat2[15] = 16;
        mat2[16] = 17;
        mat2[17] = 18;
        mat2[18] = 19;
        mat2[19] = 20;
        mat2[20] = 21;
        mat2[21] = 22;
        mat2[22] = 23;
        mat2[23] = 24;
        */

        /*
        for (int i = 0; i < mat1.Length; i++)
        {
            mat1[i] = Random.Range(-1.0f, 1.0f) * 50;
        }

        for (int i = 0; i < mat2.Length; i++)
        {
            mat2[i] = Random.Range(-1.0f, 1.0f) * 50;
        }
        

        float[] newMat = matMult(mat1, mat2); ;
        */


    }

    // Update is called once per frame
    void Update()
    {

    }

    int getIndex(int wIndex, int hIndex, int width)
    {
        return hIndex * width + wIndex;
    }

    float[] matMult(float[] mat1, float[] mat2)
    {
        Debug.Log("Hi");
        float[] newMat = new float[0];
        if (Mat1Dim.x == Mat2Dim.y)
        {
            Debug.Log("Hi 2");
            Vector2Int dim;
            if (Mat1Dim.x >= Mat2Dim.x && Mat1Dim.y >= Mat2Dim.y)
            {
                dim = Mat1Dim;
            }
            else
            {
                dim = Mat2Dim;
            }

            newMat = new float[dim.x * dim.y];


            for (int height = 0; height < dim.y; height++)
            {

                for (int width = 0; width < dim.x; width++)
                {

                    newMat[getIndex(width, height, dim.x)] = dotProduct(mat1, mat2, width, height, Mat1Dim.x);
                }
            }

            //Display the matrix
            string line = "\n";
            for (int height = 0; height < dim.y; height++)
            {
                for (int width = 0; width < dim.x; width++)
                {

                    //Debug.Log("Width: " + width + " Height: " + height + " = " + newMat[getIndex(width, height, dim.x)]);

                    line += newMat[getIndex(width, height, dim.x)] + "    ";
                }
                line += "\n";

            }
            Debug.Log(line);
        }

        return newMat;
    }

    float dotProduct(float[] mat1, float[] mat2, int wIndex, int hIndex, int length)
    {
        float result = 0;

        for (int i = 0; i < length; i++)
        {
            

            result += mat1[getIndex(i, hIndex, Mat1Dim.x)] * mat2[getIndex(wIndex, i, Mat2Dim.x)];
        }

        return result;
    }

}


