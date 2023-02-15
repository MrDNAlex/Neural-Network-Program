using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class NeuralNetwork 
{

    public Layer[] layers;

    public double inputVal;

  
    public NeuralNetwork (int[] layerSize)
    {
        layers = new Layer[layerSize.Length - 1];
        for (int i = 0; i < layerSize.Length - 1; i ++)
        {
            layers[i] = new Layer(layerSize[i], layerSize[i + 1]);
        }
    }

    public NeuralNetwork ()
    {

    }

    //Run the entire neural network and get the outputs
    double[] CalcOutput (double[] inputs)
    {
        foreach (Layer layer in layers)
        {
            inputs = layer.CalcOutput(inputs);
        }
        return inputs;
    }

    //Get the index of the output of the neural network with highest value
    public int Classify (double[] inputs)
    {
        double[] output = CalcOutput(inputs);

        return IndexOfMax(output);
    }

    public double[] Classify2 (double[] inputs)
    {
        return CalcOutput(inputs);
    }


    public int IndexOfMax (double[] inputs)
    {
        int index = 0;
        double maxVal = inputs[0];

        for (int i = 0; i < inputs.Length; i ++)
        {
            if (inputs[i] >= maxVal)
            {
                index = i;
            }
        }
        return index;
    }

    public double Cost(DataPoint dataPoint)
    {
        double[] outputs = CalcOutput(dataPoint.inputs);
        Layer outputLayer = layers[layers.Length - 1];
        double cost = 0;

        for (int nodeOut = 0; nodeOut < outputs.Length; nodeOut++)
        {
            cost += outputLayer.NodeCost(outputs[nodeOut], dataPoint.expectedOutputs[nodeOut]);
        }

        return cost;

    }

    public double Cost (DataPoint[] data)
    {
        double totalCost = 0;
        foreach(DataPoint dataPoint in data)
        {
            totalCost += Cost(dataPoint);
        }

        //Return Average
        return totalCost / data.Length;

    }

    public void Learn (DataPoint[] trainingBatch, double learnRate)
    {

        int trainingIndex = 0;

        foreach (DataPoint dataPoint in trainingBatch)
        {
            trainingIndex ++;
            UpdateAllGradients(dataPoint);
            //Debug.Log(((float)trainingIndex / trainingBatch.Length) + " % Done");
        }



        ApplyAllGradients(learnRate / trainingBatch.Length);

        //Reset All gradients

    }

    void UpdateAllGradients (DataPoint dataPoint)
    {
        //Backwards Propogation Algorithm

        //Need to get the outputVals
        CalcOutput(dataPoint.inputs);

        //Update the Final Layers Gradients
        Layer outputLayer = layers[layers.Length - 1];
        double[] nodeValues = outputLayer.CalculateOutputLayerNodeValues(dataPoint.expectedOutputs);
        outputLayer.UpdateGradients(nodeValues);

        //Update Gradients of hidden layers
        for (int hiddenLayerIndex = layers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
        {
            Layer hiddenLayer = layers[hiddenLayerIndex];
            nodeValues = hiddenLayer.CalculateHiddenNodeValues(layers[hiddenLayerIndex + 1], nodeValues);
            hiddenLayer.UpdateGradients(nodeValues);
        }

    }

    void ApplyAllGradients (double learnRate)
    {
        for (int i = 0; i < layers.Length; i ++)
        {
            layers[i].ApplyGradients(learnRate);
        }
    }

    




}


public class LayerLearnData
{
    public double[] inputs;
    public double[] weightedInputs;
    public double[] activations;
    public double[] nodeValues;

    public LayerLearnData(Layer layer)
    {
        weightedInputs = new double[layer.numNodesOut];
        activations = new double[layer.numNodesOut];
        nodeValues = new double[layer.numNodesOut];
    }

}

public class NetworkLearnData
{
    public LayerLearnData[] layerData;

    public NetworkLearnData(Layer[] layers)
    {
        layerData = new LayerLearnData[layers.Length];
        for (int i = 0; i < layers.Length; i++)
        {
            layerData[i] = new LayerLearnData(layers[i]);
        }
    }
}







