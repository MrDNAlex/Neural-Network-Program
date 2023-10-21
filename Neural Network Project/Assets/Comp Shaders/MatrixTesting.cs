using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DNAMath;
using DNANeuralNet;

public class MatrixTesting : MonoBehaviour
{
    [SerializeField] Vector2Int matrixADim;

    [SerializeField] Vector2Int matrixBDim;

    [SerializeField] int[] layerSizes;

    [SerializeField] ComputeShader shader;

    [SerializeField] List<Texture2D> images;

    List<DNAMatrix> data = new List<DNAMatrix>();

   // public DNANeuralNetworkInfo network;

    System.DateTime startTime;
    System.DateTime endTime;

    // Start is called before the first frame update
    void Start()
    {

        DNANeuralNetwork neuro = new DNANeuralNetwork(layerSizes);

        foreach (Texture2D img in images)
        {
            DNAMatrix matrix = new DNAMatrix(img.height * img.width, 1);

            for (int i = 0; i < img.height; i++)
            {
                for (int j = 0; j < img.width; j++)
                {
                    matrix[i * img.width + j, 0] = img.GetPixel(j, i).r;
                }
            }

            data.Add(matrix);
        }

        // foreach (DNAMatrix mat in data)
        // {
        //     (int index, DNAMatrix output) = neuro.Classify(mat);
        //  }



        // DNAMatrix matrixA = new DNAMatrix(matrixADim.x, matrixADim.y);
        DNAMatrix matrixA = DNAMatrix.Increment(matrixADim.x, matrixADim.y);

        DNAMatrix matrixB = DNAMatrix.Increment(matrixBDim.x, matrixBDim.y);

        startTime = System.DateTime.Now;

        //DNAMatrix result = matrixA * matrixB;
        DNAMatrix result = DNAMatrix.multMatrixGPU(matrixA, matrixB);

        endTime = System.DateTime.Now;
        Debug.Log($"Matrix Multiplication: ({matrixA.Height}x{matrixA.Width}) * ({matrixB.Height}x{matrixB.Width})");
        Debug.Log("Total Time elapsed GPU (MilliSeconds): " + (endTime - startTime).TotalMilliseconds);
        

       // result.DisplayMat();

      

       // result2.DisplayMat();


    }

    // Update is called once per frame
    void Update()
    {

    }

    int getIndex(int wIndex, int hIndex, int width)
    {
        return hIndex * width + wIndex;
    }

    float[] matMult(float[] matrixA, float[] matrixB)
    {
        Debug.Log("Hi");
        float[] newMat = new float[0];
        if (matrixADim.x == matrixBDim.y)
        {
            Debug.Log("Hi 2");
            Vector2Int dim;
            if (matrixADim.x >= matrixBDim.x && matrixADim.y >= matrixBDim.y)
            {
                dim = matrixADim;
            }
            else
            {
                dim = matrixBDim;
            }

            newMat = new float[dim.x * dim.y];


            for (int height = 0; height < dim.y; height++)
            {

                for (int width = 0; width < dim.x; width++)
                {

                    newMat[getIndex(width, height, dim.x)] = dotProduct(matrixA, matrixB, width, height, matrixADim.x);
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

    float dotProduct(float[] matrixA, float[] matrixB, int wIndex, int hIndex, int length)
    {
        float result = 0;

        for (int i = 0; i < length; i++)
        {
            

            result += matrixA[getIndex(i, hIndex, matrixADim.x)] * matrixB[getIndex(wIndex, i, matrixBDim.x)];
        }

        return result;
    }

}


