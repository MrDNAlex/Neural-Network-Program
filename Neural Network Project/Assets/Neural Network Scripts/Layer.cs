using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer
{
    //Input layer doesn't count


    //Eventually replace weight stuff with a vector representation? Maybe custom library

    public int numNodesIn, numNodesOut;

    public double[] costGradientBias;
    public double[,] costGradientWeights;

    public double[] biases;
    public double[,] weights;

    public double[] activations;
    public double[] weightedInputs;
    public double[] inputs;


    //NodesIn = number of neurons the previous layer had
    //NodesOut = number of neurons of this layer

    public Layer (int nodesIn, int nodesOut)
    {
        numNodesIn = nodesIn;
        numNodesOut = nodesOut;

        weights = new double[nodesIn, nodesOut];
        biases = new double[nodesOut];
        costGradientWeights = new double[nodesIn, nodesOut];
        costGradientBias = new double[nodesOut];

        InitRandomWeights();
    }


    //Calculate the output of the layer
    public double [] CalcOutput (double [] input)
    {
        inputs = input;

        double[] activations = new double[numNodesOut];
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut ++)
        {
            double weightInput = biases[nodeOut];
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                weightInput += input[nodeIn] * weights[nodeIn, nodeOut];
            }

            weightedInputs[nodeOut] = weightInput;
            activations[nodeOut] = ActivationFunction(weightInput);
        }

        this.activations = activations;

        return activations;

    }

    double ActivationFunction(double weightedInput)
    {
        //Sigmoid
        return 1 / (1 + Mathf.Exp(-(float)weightedInput));

        /*
        //Hyperbolic tangent
        double e2w = Mathf.Exp(2 * (float)weightedInput);
        return (e2w - 1) / (e2w + 1);

        //SiLU
        return weightedInput / (1 + Mathf.Exp(-(float)weightedInput));

        //ReLU
        return Mathf.Max(0, weightedInput)
        */
    }

    public double NodeCost (double outputVal, double expectedVal)
    {
        double error = outputVal - expectedVal;
        return error * error;
    }


    //make a function to give random initial weights 
  
    public void ApplyGradients (double learnRate)
    {
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
        {
            biases[nodeOut] -= costGradientBias[nodeOut] * learnRate;

            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
            {
                weights[nodeIn, nodeOut] -= costGradientWeights[nodeIn, nodeOut] * learnRate;
            }


        }

    }


    public void InitRandomWeights ()
    {
        //Apply random vals to all weights 

        System.Random rng = new System.Random();

        for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn ++)
        {
            for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
            {
                //Get a value between -1 and 1
                double ranVal = rng.NextDouble() * 2 - 1;

                weights[nodeIn, nodeOut] = ranVal / Mathf.Sqrt(numNodesIn);

            }
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

    public double[] CalculateHiddenNodeValues (Layer oldLayer, double[] oldNodeVals)
    {
        double[] newNodeVals = new double[numNodesOut];
        for (int newNodeIndex = 0; newNodeIndex < newNodeVals.Length; newNodeIndex++)
        {
            double newNodeVal = 0;

            for (int oldNodeIndex = 0; oldNodeIndex < oldNodeVals.Length; oldNodeIndex ++)
            {
                double weightedInputDeriv = oldLayer.weights[newNodeIndex, oldNodeIndex];
                newNodeVal += weightedInputDeriv * oldNodeVals[oldNodeIndex];
            }

            newNodeVal *= ActivationDerivative(weightedInputs[newNodeIndex]);
            newNodeVals[newNodeIndex] = newNodeVal;

        }
        return newNodeVals;

    }

    public void UpdateGradients (double[] nodeVals)
    {
       
        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut ++)
        {
            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn ++)
            {
                //Update gradients for weights
                double derivativeCostWRTWeight = inputs[nodeIn] * nodeVals[nodeOut];

                costGradientWeights[nodeIn, nodeOut] += derivativeCostWRTWeight;
            }

            //Update Gradient for biases
            double derivativeCostWRTBias = 1 * nodeVals[nodeOut];
            costGradientBias[nodeOut] += derivativeCostWRTBias;
        }

    }


   
    











}
