using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class Layer
{
    //Input layer doesn't count


    //Eventually replace weight stuff with a vector representation? Maybe custom library

    public int numNodesIn, numNodesOut;

    public double[] costGradientBias;
    // public double[,] costGradientWeights;
    public double[] costGradientWeights;

    public double[] biases;
    // public double[,] weights;
    public double[] weights;

    public double[] weightVolicities;
    public double[] biasVelocities;

    

    public double[] activations;
    public double[] weightedInputs;
    public double[] inputs;


    //NodesIn = number of neurons the previous layer had
    //NodesOut = number of neurons of this layer

    public Layer (int nodesIn, int nodesOut)
    {
        numNodesIn = nodesIn;
        numNodesOut = nodesOut;

        weights = new double[nodesIn* nodesOut];
        biases = new double[nodesOut];

        costGradientWeights = new double[weights.Length];
        costGradientBias = new double[biases.Length];

        weightVolicities = new double[weights.Length];
        biasVelocities = new double[biases.Length];


        weightedInputs = new double[nodesOut];

        InitRandomWeights();
    }

    //Might not need this anymore
    public Layer (LayerSaver laySave)
    {

        int nodesIn = laySave.numNodesIn;
        int nodesOut = laySave.numNodesOut;

        numNodesIn = nodesIn;
        numNodesOut = nodesOut;

        weights = new double[nodesIn* nodesOut];
        biases = new double[nodesOut];
        costGradientWeights = new double[nodesIn* nodesOut];
        costGradientBias = new double[nodesOut];

        weightedInputs = new double[nodesOut];

        this.biases = laySave.biases;

        //load weights 
        for (int i = 0; i < numNodesIn * numNodesOut; i++)
        {
                weights[i] = laySave.weights[i];
        }
    }

    public double[] CalcOutput(double[] input)
    {

        inputs = input;

        double[] activations = new double[numNodesOut];
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            double weightInput = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {

                weightInput += input[nodeIn] * getWeight(nodeIn, nodeOut);
            }

            weightedInputs[nodeOut] = weightInput;
            activations[nodeOut] = ActivationFunction(weightInput);
        }

        this.activations = activations;

        return activations;

    }

    //Calculate the output of the layer
    public double [] CalcOutput (double [] input, LayerLearnData learnData)
    {
        learnData.inputs = input;

       // inputs = input;

        //double[] activations = new double[numNodesOut];
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut ++)
        {
            double weightInput = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
               
                weightInput += input[nodeIn] * getWeight(nodeIn, nodeOut);
            }

            learnData.weightedInputs[nodeOut] = weightInput;

           // weightedInputs[nodeOut] = weightInput;
           // activations[nodeOut] = ActivationFunction(weightInput);
        }

        //Apply Activation Function 
        for (int i = 0; i < learnData.activations.Length; i ++)
        {
            learnData.activations[i] = ActivationFunction(learnData.weightedInputs, i);
        }

       // this.activations = activations;

        return learnData.activations;

    }

    double ActivationFunction(double weightedInput)
    {
        //Sigmoid
        return 1 / (1 + Mathf.Exp(-(float)weightedInput));

        
        //Hyperbolic tangent
       // double e2w = Mathf.Exp(2 * (float)weightedInput);
       // return (e2w - 1) / (e2w + 1);

        //SiLU
       // return weightedInput / (1 + Mathf.Exp(-(float)weightedInput));

        //ReLU
       // return Mathf.Max(0, (float)weightedInput);
        
    }

    double ActivationFunction(double[] weightedInput, int index)
    {
        //Sigmoid
        return 1 / (1 + Mathf.Exp(-(float)weightedInput[index]));


        //Hyperbolic tangent
        // double e2w = Mathf.Exp(2 * (float)weightedInput);
        // return (e2w - 1) / (e2w + 1);

        //SiLU
        // return weightedInput / (1 + Mathf.Exp(-(float)weightedInput));

        //ReLU
        //return Mathf.Max(0, (float)weightedInput[index]);

    }



    public double NodeCost (double outputVal, double expectedVal)
    {
        double error = outputVal - expectedVal;
        return error * error;
    }

    //Update weights and biases
    public void ApplyGradients (double learnRate, double regularization, double momentum)
    {
        double weightDecay = (1 - regularization * learnRate);

        for (int i = 0; i < weights.Length; i ++)
        {
            double weight = weights[i];
            double velocity = weightVolicities[i] * momentum - costGradientWeights[i] * learnRate;
            weightVolicities[i] = velocity;
            weights[i] = weight * weightDecay + velocity;
            costGradientWeights[i] = 0;
        }

        for (int i = 0; i < biases.Length; i ++)
        {
            double velocity = biasVelocities[i] * momentum - costGradientBias[i] * learnRate;
            biasVelocities[i] = velocity;
            biases[i] += velocity;
            costGradientBias[i] = 0;
        }

        /*
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            biases[nodeOut] -= costGradientBias[nodeOut] * learnRate;

            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                weights[nodeIn, nodeOut] -= costGradientWeights[nodeIn, nodeOut] * learnRate;
            }


        }


        //Reset Gradients

        for (int i = 0; i < costGradientBias.Length; i ++)
        {
            costGradientBias[i] = 0;
        }

        for (int i = 0; i < numNodesIn; i++)
        {
            for (int j = 0; j < numNodesOut; j ++)
            {
                costGradientWeights[i, j] = 0;
            }
            
        }
        */

    }


    public void InitRandomWeights ()
    {
        //Apply random vals to all weights 

        System.Random rng = new System.Random();

        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = RandomInNormalDistribution(rng, 0, 1) / Mathf.Sqrt(numNodesIn);
        }

        double RandomInNormalDistribution(System.Random rng, double mean, double standardDeviation)
        {
            double x1 = 1 - rng.NextDouble();
            double x2 = 1 - rng.NextDouble();

            double y1 = Mathf.Sqrt(-2.0f * Mathf.Log((float)x1)) * Mathf.Cos(2.0f * (float)Mathf.PI * (float)x2);
            return y1 * standardDeviation + mean;
        }
    }


    //
    //Derivatives
    //
    public double nodeCostDerivative (double outputVal, double expectedVal)
    {
        return 2 * (outputVal - expectedVal);
    }

    public double ActivationDerivative (double weightedVal)
    {
        //Sigmoid
        double activationVal = ActivationFunction(weightedVal);
       return activationVal * (1 - activationVal);

        //ReLU
     //   return (weightedVal > 0) ? 1 : 0;
    }

    public double ActivationDerivative(double[] weightedVal, int index)
    {
        //Sigmoid
        double activationVal = ActivationFunction(weightedVal[index]);
        return activationVal * (1 - activationVal);

        //ReLU
        //return (weightedVal[index] > 0) ? 1 : 0;
    }

    public double weightedValDerivative ()
    {
        //Value of the last activation function

        return 2;
    }


    public double[] CalculateOutputLayerNodeValues (double[] expectedOutputs)
    {
        double[] nodeVals = new double[expectedOutputs.Length];
        for (int i = 0; i < nodeVals.Length; i ++)
        {

            double costDerivative = nodeCostDerivative(activations[i], expectedOutputs[i]);
            double activationDerivative = ActivationDerivative(weightedInputs[i]);
            nodeVals[i] = costDerivative * activationDerivative;
        }
        return nodeVals;
    }

    //Seb Lague
    public double[] CalculateOutputLayerNodeValues(LayerLearnData layerData, double[] expectedOutputs)
    {
        double[] nodeVals = new double[expectedOutputs.Length];
        for (int i = 0; i < nodeVals.Length; i++)
        {

            double costDerivative = nodeCostDerivative(layerData.activations[i], expectedOutputs[i]);
            double activationDerivative = ActivationDerivative(layerData.weightedInputs, i);
            layerData.nodeValues[i] = costDerivative * activationDerivative;
        }
        return nodeVals;
    }

    public double[] CalculateHiddenNodeValues (Layer oldLayer, double[] oldNodeVals)
    {
        double[] newNodeVals = new double[numNodesOut];
        for (int newNodeIndex = 0; newNodeIndex < newNodeVals.Length; newNodeIndex++)
        {
            double newNodeVal = 0;

            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeVals.Length; oldNodeIndex ++)
            {
                double weightedInputDeriv = oldLayer.getWeight(newNodeIndex, oldNodeIndex);
                newNodeVal += weightedInputDeriv * oldNodeVals[oldNodeIndex];
            }

            newNodeVal *= ActivationDerivative(weightedInputs[newNodeIndex]);
            newNodeVals[newNodeIndex] = newNodeVal;

        }
        return newNodeVals;

    }

    //Seb Lague
    public void CalculateHiddenNodeValues(LayerLearnData learnData, Layer oldLayer, double[] oldNodeVals)
    {

        for (int newNodeIndex = 0; newNodeIndex < numNodesOut; newNodeIndex++)
        {
            double newNodeVal = 0;
            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeVals.Length; oldNodeIndex++)
            {
                double weightedInputDeriv = oldLayer.getWeight(newNodeIndex, oldNodeIndex);
                newNodeVal += weightedInputDeriv * oldNodeVals[oldNodeIndex];
            }

            newNodeVal *= ActivationDerivative(learnData.weightedInputs, newNodeIndex);
            learnData.nodeValues[newNodeIndex] = newNodeVal;
        }
    }


    public void UpdateGradients (double[] nodeVals)
    {
       
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut ++)
        {
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn ++)
            {
                //Update gradients for weights
                double derivativeCostWRTWeight = inputs[nodeIn] * nodeVals[nodeOut];

                costGradientWeights[getFlatWeightIndex(nodeIn, nodeOut)] += derivativeCostWRTWeight;
            }

            //Update Gradient for biases
            double derivativeCostWRTBias = 1 * nodeVals[nodeOut];
            costGradientBias[nodeOut] += derivativeCostWRTBias;
        }

    }

    //Seb Lague
    public void UpdateGradients(LayerLearnData layerData)
    {

        //Update Weights (lock for multithreading)
        
        lock (costGradientWeights)
        {
            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
            {
                double nodeVal = layerData.nodeValues[nodeOut];
                for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
                {
                    double derivativeCostWRTWeight = layerData.inputs[nodeIn] * nodeVal;

                    costGradientWeights[getFlatWeightIndex(nodeIn, nodeOut)] += derivativeCostWRTWeight;

                }
            }
        }

        //Update cost gradient Bias (lock for multithreading)
        lock (costGradientBias)
        {

            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
            {
                double derivativeCostWRTBias = 1 * layerData.nodeValues[nodeOut];
                costGradientBias[nodeOut] += derivativeCostWRTBias;
            }
        }

    }

    public double getWeight (int nodeIn, int nodeOut)
    {

        return weights[nodeOut * numNodesIn + nodeIn];


    }

    public int getFlatWeightIndex (int inputNeurIndex, int outputNeurIndex)
    {
        return outputNeurIndex * numNodesIn + inputNeurIndex;

    }


}
