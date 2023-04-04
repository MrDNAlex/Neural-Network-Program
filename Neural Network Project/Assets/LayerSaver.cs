using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DNANeuralNet;

[System.Serializable]


public class LayerSaver 
{

    public double[] weights;

    public int numNodesIn, numNodesOut;

    //public double[] costGradientBias;
   // public double[,] costGradientWeights;

    public double[] biases;
   // public double[,] weights;

   // List<List<double>> weights = new List<List<double>>();

    

    

   // public double[] activations;
   // public double[] weightedInputs;
    //public double[] inputs;


    //Just need to change the format of weights and costGradientWeights. Just weights

    public LayerSaver (Layer layer)
    {
        this.numNodesIn = layer.numNodesIn;
        this.numNodesOut = layer.numNodesOut;
        this.biases = layer.biases;


        weights = new double[numNodesIn * numNodesOut];
        //Convert weights to a friendlier format

        for (int i = 0; i < weights.Length; i ++)
        {
           
                weights[i] = layer.weights[i]; 
           
        }


    }


}
