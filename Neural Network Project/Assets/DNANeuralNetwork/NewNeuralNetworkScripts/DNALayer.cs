using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using DNAMath;

namespace DNANeuralNet
{
    //Maybe we make this into a interface
    [System.Serializable]
    public class DNALayer
    {
        public ILayer iLayer;

        //Maybe make biases a matrix?
        public DNAMatrix[] weights;
        public double[] biases;

        public DNAMatrix[] costGradientWeight;
        public double[] costGradientBias;

        //Momentum
        public DNAMatrix[] weightVelocities;
        public double[] biasVelocities;


        public Vector2Int outputSize;
        public int outputMatNum;

        public double GetWeight(int nodeIn, int nodeOut, int matrixIndex)
        {
            int flatIndex = nodeOut * weights[matrixIndex].matrixDimensions.x + nodeIn;
            return weights[matrixIndex].values[flatIndex];
        }

        public int GetFlatWeightIndex(int xIndex, int yIndex)
        {
            return yIndex * weights[0].matrixDimensions.x + xIndex;
        }

        public void InitializeRandomWeights(System.Random rng)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].matrixDimensions.x * weights[i].matrixDimensions.y; j++)
                {
                    weights[i].values[j] = RandomInNormalDistribution(rng, 0, 1) / Mathf.Sqrt((float)weights[i].values.Length);
                }
            }

            double RandomInNormalDistribution(System.Random rng, double mean, double standardDeviation)
            {
                double x1 = 1 - rng.NextDouble();
                double x2 = 1 - rng.NextDouble();

                double y1 = Mathf.Sqrt(-2.0f * Mathf.Log((float)x1)) * Mathf.Cos(2.0f * Mathf.PI * (float)x2);
                return y1 * standardDeviation + mean;
            }
        }

        public DNAMatrix[] CalculateOutputs(DNAMatrix[] inputs, DNALayerLearnData layerLearnData)
        {
            layerLearnData.inputs = inputs;

            DNAMatrix[] outputs = iLayer.CalculateOutputs(inputs);

            layerLearnData.outputs = outputs;

            return outputs;
        }

        public void CalculateOutputLayerNodeValues(DNALayerLearnData layerLearnData, DNAMatrix expectedOutputs, ICost cost)
        {
            layerLearnData.nodeValues = new DNAMatrix[1];
            layerLearnData.nodeValues[0] = new DNAMatrix(expectedOutputs.matrixDimensions);
            for (int i = 0; i < expectedOutputs.values.Length; i++)
            {
                layerLearnData.nodeValues[0].values[i] = cost.CostDerivative(layerLearnData.outputs[0].values[i], expectedOutputs.values[i]);
            }
        }


        //Functions for 

        //Update Gradients
        //CalculateHiddenLayerNodeValues
        //CalculateOutputLayerNodeValues   X
        //Apply gradients
        //Calculate outputs (Returns the matrices)  X X


        //Calculate output Node values   (This one needs to look if it needs to unflatten, and then basically take the values

        //Calculate input node values    (This is where we will apply the current layers changes




        //Maybe look at this 
        //https://towardsdatascience.com/backpropagation-in-fully-convolutional-networks-fcns-1a13b75fb56a#:~:text=Backpropagation%20has%20two%20phases.,a%20few%20Momentum%2C%20Adam%20etc%20%E2%80%A6


    }

    [System.Serializable]
    public class NeuralLayer : DNALayer, ILayer
    {
        public int numNodesIn;
        public int numNodesOut;


        public NeuralLayer(int numNodesIn, int numNodesOut)
        {
            this.iLayer = this;
            this.numNodesIn = numNodesIn;
            this.numNodesOut = numNodesOut;

            Debug.Log("Nodes in: " + numNodesIn);

            //Generate number needed for all
            weights = new DNAMatrix[1];
            costGradientWeight = new DNAMatrix[1];
            weightVelocities = new DNAMatrix[1];

            // for (int i = 0; i < 1; i++)
            // {
            weights[0] = new DNAMatrix(new Vector2Int(numNodesIn, numNodesOut));
            // }

            biases = new double[numNodesOut];
            costGradientBias = new double[numNodesOut];
            biasVelocities = new double[numNodesOut];

            InitializeRandomWeights(new System.Random());

            this.outputSize = new Vector2Int(1, numNodesOut);

            this.outputMatNum = 1;
            // Debug.Log(outputSize);
        }

        public Vector2Int getOutputSize(Vector2Int inputSize)
        {
            return new Vector2Int(1, numNodesOut);
        }

        public int flattenLayer(Vector2Int inputtedSize)
        {
            Debug.Log("Hi");
            return numNodesOut;
        }

        public DNAMatrix[] CalculateOutputs(DNAMatrix[] inputs)
        {
            //Flatten the matrices and then do the matrix multiplication
            DNAMatrix flattenedMatrix = new DNAMatrix(new Vector2Int(1, numNodesIn));

            List<double> vals = new List<double>();
            foreach (DNAMatrix mat in inputs)
            {
                foreach (double val in mat.values)
                {
                    vals.Add(val);
                }
            }

            flattenedMatrix.values = vals.ToArray();

            DNAMatrix outputMat = new DNAMatrix(new Vector2Int(1, numNodesOut));

            //Multiply the matrices
            outputMat = flattenedMatrix * weights[0];

            //Add biases
            for (int i = 0; i < outputMat.values.Length; i++)
            {
                outputMat.values[i] = outputMat.values[i] + biases[i];
            }

            DNAMatrix[] outputs = new DNAMatrix[1];

            outputs[0] = outputMat;

            return outputs;
        }

        public void CalculateHiddenLayerOutputNodeValues (DNALayerLearnData layerLearnData, DNAMatrix[] oldNodeValues)
        {
            //In the event that we need to unflatten

            int length = oldNodeValues.Length * oldNodeValues[0].matrixDimensions.x * oldNodeValues[0].matrixDimensions.y;

            DNAMatrix flatten = new DNAMatrix(new Vector2Int(1, length));

            for (int i = 0; i < oldNodeValues.Length; i ++)
            {
                for (int height = 0; height < oldNodeValues[i].matrixDimensions.x; height++)
                {
                    for (int width = 0; width < oldNodeValues[i].matrixDimensions.y; width++)
                    {
                        flatten.setValue(0, height * oldNodeValues[i].matrixDimensions.y + width, oldNodeValues[i].getValue(height, width));
                    }
                }
            }

            layerLearnData.nodeValues = new DNAMatrix[1];

            layerLearnData.nodeValues[0] = flatten;
        }

        public void CalculateHiddenLayerInputNodeValues (DNALayerLearnData layerLearnData)
        {
            //Won't need to flatten

            for (int i = 0; i < layerLearnData.inputs.Length; i ++)
            {
                DNAMatrix inputMatrix = new DNAMatrix(layerLearnData.inputs[i].matrixDimensions);


                //We know it will only have 




            }







        }



        //I Honestly don't know if this is it
        public void CalculateHiddenLayerNodeValues (DNALayerLearnData layerLearnData, DNALayer oldLayer, DNAMatrix[] oldNodeValues)
        {
            for (int i = 0; i < numNodesOut; i++)
            {
                double newNodeVal = 0;
                for (int j = 0; j < oldNodeValues.Length; j ++)
                {
                    for (int height = 0; height < oldNodeValues[j].matrixDimensions.x; height ++)
                    {
                        for (int width = 0; width < oldNodeValues[j].matrixDimensions.y; width++)
                        {
                            newNodeVal += oldLayer.GetWeight(i, height * oldNodeValues[j].matrixDimensions.y + width, j);
                        }
                    }
                }
                layerLearnData.nodeValues[0].setValue(0, i, newNodeVal);

            }


            //needs to take in last layers node values and set the node values of this one to basically the output size
        }



    }

    [System.Serializable]
    public class PoolingLayer : DNALayer, ILayer
    {
        Vector2Int poolSize;
        int stride;
        PoolingType poolType;
        //Max, Average or Min pooling

        //Add type pooling
        public PoolingLayer(Vector2Int poolSize, int stride, PoolingType poolType, Vector2Int lastSize, int filterNum = 1)
        {
            iLayer = this;
            this.poolSize = poolSize;
            this.stride = stride;
            this.poolType = poolType;

            weights = new DNAMatrix[filterNum];

            this.outputSize = getOutputSize(lastSize);

            //Debug.Log(outputSize);

            this.outputMatNum = filterNum;

            Debug.Log("Pool Layer: Size:" + poolSize + " Num: " + filterNum + " Stride: " + stride + "Output:" + outputSize);
        }

        public Vector2Int getOutputSize(Vector2Int inputtedSize)
        {

            //Width
            int width = 0;
            for (int i = 0; i < inputtedSize.x; i = i + stride)
            {
                width++;

                //Actually, just don't cover the last region
                if (i + (poolSize.x - 1) + stride >= inputtedSize.x)
                {
                    i = inputtedSize.x;
                }
            }

            //Height
            int height = 0;
            for (int i = 0; i < inputtedSize.y; i = i + stride)
            {
                height++;

                //Actually, just don't cover the last region
                if (i + (poolSize.y - 1) + stride >= inputtedSize.y)
                {
                    i = inputtedSize.y;
                }
            }

            Vector2Int outputtedSize = new Vector2Int(width, height);

            return outputtedSize;
        }

        public int flattenLayer(Vector2Int inputtedSize)
        {
            Vector2Int size = getOutputSize(inputtedSize);

            return weights.Length * size.x * size.y;
        }

        public DNAMatrix[] CalculateOutputs(DNAMatrix[] inputs)
        {
            DNAMatrix[] outputs = new DNAMatrix[inputs.Length];

            for (int i = 0; i < inputs.Length; i++)
            {
                //Apply pooling to each 
                outputs[i] = new DNAMatrix(outputSize);

                for (int yIndex = 0; yIndex < outputSize.y; yIndex++)
                {
                    for (int xIndex = 0; xIndex < outputSize.x; xIndex++)
                    {
                        //Get the current index, multiply each by the stride to get correct start index, then pass 

                        int trueYIndex = yIndex * stride;
                        int trueXIndex = xIndex * stride;

                        outputs[i].setValue(yIndex, xIndex, getPoolValue(trueYIndex, trueXIndex, inputs[i]));

                    }
                }
            }

            return outputs;
        }

        double getPoolValue(int yIndex, int xIndex, DNAMatrix matrix)
        {
            //  Debug.Log(matrix.matrixDimensions);

            //  Debug.Log(new Vector2Int(yIndex, xIndex));

            //Debug.Log(matrix);

            double output = matrix.getValue(yIndex, xIndex);
            switch (poolType)
            {
                case PoolingType.Max:
                    //Get the maximum value
                    for (int y = 0; y < stride; y++)
                    {
                        for (int x = 0; x < stride; x++)
                        {
                            double val = matrix.getValue(yIndex + y, xIndex + x);
                            if (val > output)
                            {
                                output = val;
                            }
                        }
                    }
                    break;
                case PoolingType.Min:
                    //Get the Minimum value
                    for (int y = 0; y < stride; y++)
                    {
                        for (int x = 0; x < stride; x++)
                        {
                            double val = matrix.getValue(yIndex + y, xIndex + x);
                            if (val < output)
                            {
                                output = val;
                            }
                        }
                    }
                    break;
                case PoolingType.Average:
                    //Get the average value
                    double total = 0;
                    for (int y = 0; y < stride; y++)
                    {
                        for (int x = 0; x < stride; x++)
                        {
                            total += matrix.getValue(yIndex + y, xIndex + x);
                        }
                    }

                    output = total / (stride * stride);
                    break;
                default:
                    //Default to Max
                    //Get the maximum value
                    for (int y = 0; y < stride; y++)
                    {
                        for (int x = 0; x < stride; x++)
                        {
                            double val = matrix.getValue(yIndex + y, xIndex + x);
                            if (val > output)
                            {
                                output = val;
                            }
                        }
                    }
                    break;
            }

            return output;
        }

        public void CalculateHiddenLayerNodeValues(DNALayerLearnData layerLearnData, DNALayer oldLayer, DNAMatrix[] oldNodeValues)
        {
            for (int i = 0; i < outputMatNum; i ++)
            {

            }
        }

    }

    [System.Serializable]
    public class FilterLayer : DNALayer, ILayer
    {

        //Gonna have to fix this
        //The number of outputted layers is equal to the number of filters. we need to add up the values from 2 layers sharing a bias



        //Variable for input size?

        int stride;
        Vector2Int filterSize;
        int filterNum;
        int lastLayNum;

        //Last layer num is the value of filternum from the last layer
        public FilterLayer(Vector2Int filterSize, int filterNum, int stride, Vector2Int lastSize, int lastLayerNum = 1)
        {
            this.iLayer = this;
            this.filterSize = filterSize;
            this.stride = stride;
            this.filterNum = filterNum;

            //Generate number needed for all
            weights = new DNAMatrix[filterNum * lastLayerNum];
            costGradientWeight = new DNAMatrix[filterNum * lastLayerNum];
            weightVelocities = new DNAMatrix[filterNum * lastLayerNum];

            for (int i = 0; i < filterNum * lastLayerNum; i++)
            {
                weights[i] = new DNAMatrix(filterSize);
            }

            biases = new double[filterNum];
            costGradientBias = new double[filterNum];
            biasVelocities = new double[filterNum];

            lastLayNum = lastLayerNum;

            InitializeRandomWeights(new System.Random());

            this.outputSize = getOutputSize(lastSize);

            this.outputMatNum = filterNum;

            //Debug.Log(outputSize);

            Debug.Log("Filter Layer: Size:" + filterSize + " Num: " + filterNum + " Stride: " + stride + "Output:" + outputSize);

        }

        //Maybe filterSize and stride can be taken from when the class is created
        public Vector2Int getOutputSize(Vector2Int inputtedSize)
        {
            //Width
            int width = 0;
            for (int i = 0; i < inputtedSize.x; i = i + stride)
            {
                width++;

                //Actually, just don't cover the last region
                if (i + (filterSize.x - 1) + stride >= inputtedSize.x)
                {
                    i = inputtedSize.x;
                }
            }

            //Height
            int height = 0;
            for (int i = 0; i < inputtedSize.y; i = i + stride)
            {
                height++;

                //Actually, just don't cover the last region
                if (i + (filterSize.y - 1) + stride >= inputtedSize.y)
                {
                    i = inputtedSize.y;
                }
            }

            Vector2Int outputtedSize = new Vector2Int(width, height);

            return outputtedSize;
        }

        public int flattenLayer(Vector2Int inputtedSize)
        {
            Debug.Log("Hi");

            Vector2Int size = getOutputSize(inputtedSize);

            return (weights.Length/lastLayNum) * size.x * size.y;
        }

        public DNAMatrix[] CalculateOutputs(DNAMatrix[] inputs)
        {
            DNAMatrix[] outputs = new DNAMatrix[filterNum];
            for (int j = 0; j < filterNum; j++)
            {
                outputs[j] = new DNAMatrix(outputSize);

                for (int i = 0; i < inputs.Length; i++)
                {
                    for (int yIndex = 0; yIndex < outputSize.y; yIndex++)
                    {
                        for (int xIndex = 0; xIndex < outputSize.x; xIndex++)
                        {
                            //Get the current index, multiply each by the stride to get correct start index, then pass 

                            int trueYIndex = yIndex * stride;
                            int trueXIndex = xIndex * stride;

                            outputs[j].addValue(yIndex, xIndex, getFilterVal(trueYIndex, trueXIndex, inputs[i], j) + (biases[j]/inputs.Length));
                        }
                    }
                }
            }

            return outputs;
        }

        public double getFilterVal(int yIndex, int xIndex, DNAMatrix matrix, int filterIndex)
        {
            double total = 0;
            for (int y = 0; y < stride; y++)
            {
                for (int x = 0; x < stride; x++)
                {
                    total += matrix.getValue(yIndex + y, xIndex + x) * weights[filterIndex].getValue(y, x);
                }
            }

            return total;
        }


        //

        //Array of matrices for each filter

        //Array of same length for biases


    }

    [System.Serializable]
    public class ActivationLayer : DNALayer, ILayer
    {
        public IActivation activation;

        //When inputed just go every value in the matrix and apply activation, then pass on to next layer without changing much

        public ActivationLayer(Activation.ActivationType type, Vector2Int lastSize)
        {
            switch (type)
            {
                case Activation.ActivationType.Sigmoid:
                    this.activation = new Activation.Sigmoid();
                    break;
                case Activation.ActivationType.ReLU:
                    this.activation = new Activation.ReLU();
                    break;
                case Activation.ActivationType.SiLU:
                    this.activation = new Activation.SiLU();
                    break;
                case Activation.ActivationType.TanH:
                    this.activation = new Activation.TanH();
                    break;
                case Activation.ActivationType.Softmax:
                    this.activation = new Activation.Softmax();
                    break;
            }

            this.outputSize = getOutputSize(lastSize);



            //  Debug.Log(outputSize);

        }

        public Vector2Int getOutputSize(Vector2Int inputSize)
        {
            return inputSize;
        }

        public int flattenLayer(Vector2Int inputtedSize)
        {
            Debug.Log("Hi");

            Vector2Int size = getOutputSize(inputtedSize);

            return weights.Length * size.x * size.y;
        }

        public DNAMatrix[] CalculateOutputs(DNAMatrix[] inputs)
        {
            DNAMatrix[] outputs = new DNAMatrix[inputs.Length];

            for (int i = 0; i < inputs.Length; i++)
            {
                for (int yIndex = 0; yIndex < outputSize.y; yIndex++)
                {
                    for (int xIndex = 0; xIndex < outputSize.x; xIndex++)
                    {

                        outputs[i].setValue(yIndex, xIndex, getActivationValue(outputs[i].getValue(yIndex, xIndex)));

                    }
                }
            }

            return outputs;

        }

        public double getActivationValue(double value)
        {
            double val = value;


            return val;
        }
    }

    public interface ILayer
    {
        public Vector2Int getOutputSize(Vector2Int inputSize);

        public int flattenLayer(Vector2Int inputtedSize);

        public DNAMatrix[] CalculateOutputs(DNAMatrix[] inputs);
    }



}

