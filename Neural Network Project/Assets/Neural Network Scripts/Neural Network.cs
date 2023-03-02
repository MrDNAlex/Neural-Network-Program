using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class NeuralNetwork 
{

    public Layer[] layers;
    public double inputVal;
    NetworkLearnData[] batchLearnData;

   
  
    public NeuralNetwork (int[] layerSize, Activation hidden, Activation output)
    {
        layers = new Layer[layerSize.Length - 1];
        for (int i = 0; i < layerSize.Length - 1; i ++)
        {
            if (i == layerSize.Length - 2)
            {
                layers[i] = new Layer(layerSize[i], layerSize[i + 1], output);
            } else
            {
                layers[i] = new Layer(layerSize[i], layerSize[i + 1], hidden);
            }
           
        }
    }

    public NeuralNetwork ()
    {

    }

    //Run the entire neural network and get the outputs
    double[] CalculateOutput (double[] inputs)
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
        double[] output = CalculateOutput(inputs);

        return IndexOfMax(output);
    }

    public double[] Classify2 (double[] inputs)
    {
        return CalculateOutput(inputs);
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
        double[] outputs = CalculateOutput(dataPoint.inputs);
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

    public void Learn (DataPoint[] trainingBatch, double learnRate, double regularization = 0, double momentum = 0)
    {

        // int trainingIndex = 0;
        if (batchLearnData == null || batchLearnData.Length != trainingBatch.Length)
        {
            batchLearnData = new NetworkLearnData[trainingBatch.Length];

            for (int i = 0; i < batchLearnData.Length; i++)
            {
                batchLearnData[i] = new NetworkLearnData(layers);
            }
        }
           

        System.Threading.Tasks.Parallel.For(0, trainingBatch.Length, (i) =>
        {
            // trainingIndex++;
            UpdateAllGradients(trainingBatch[i], batchLearnData[i]);
        });

        for (int i = 0; i < layers.Length; i++)
        {
            layers[i].ApplyGradients(learnRate / trainingBatch.Length, regularization, momentum);
        }


        //Reset All gradients

    }

    void UpdateAllGradients (DataPoint dataPoint, NetworkLearnData learnData)
    {
        //Backwards Propogation Algorithm

        double[] inputsToNextLayer = dataPoint.inputs;

        for (int i = 0; i <  layers.Length; i ++)
        {
            inputsToNextLayer = layers[i].CalcOutput(inputsToNextLayer, learnData.layerData[i]);
        }


        // -- Backpropogation --
        int outputLayerIndex = layers.Length - 1;
        Layer outputLayer = layers[outputLayerIndex];
        LayerLearnData outputLearnData = learnData.layerData[outputLayerIndex];

        //Update output layer gradients
        outputLayer.CalculateOutputLayerNodeValues(outputLearnData, dataPoint.expectedOutputs);
        outputLayer.UpdateGradients(outputLearnData);


        //Update Hidden layer
        for (int i = outputLayerIndex - 1; i >= 0; i --)
        {
            LayerLearnData layerData = learnData.layerData[i];
            Layer hiddenLayer = layers[i];

            hiddenLayer.CalculateHiddenNodeValues(layerData, layers[i + 1], learnData.layerData[i + 1].nodeValues);
            hiddenLayer.UpdateGradients(layerData);
        }


        //Need to get the outputVals
        // CalculateOutput(dataPoint.inputs);

        /*
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
        */

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









