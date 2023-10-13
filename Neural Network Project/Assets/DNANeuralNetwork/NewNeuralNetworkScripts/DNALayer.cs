using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using DNAMath;

namespace DNANeuralNet
{
    public class DNALayer
    {
        public int NumNodesIn { get; private set; }
        public int NumNodesOut { get; private set; }

        public DNAMatrix weights;
        public DNAMatrix biases;

        //Cost Gradient With respect to weight and biases
        private DNAMatrix _costGradientWeight;
        private DNAMatrix _costGradientBias;

        //Momentum
        private DNAMatrix _weightVelocities;
        private DNAMatrix _biasVelocities;

        public IActivation activation;


        public DNALayer (int numNodesIn, int numNodesOut)
        {
            this.NumNodesIn = numNodesIn;
            this.NumNodesOut = numNodesOut;
            activation = new Activation.Sigmoid();

            weights = new DNAMatrix(numNodesOut, numNodesIn);
            _costGradientWeight = new DNAMatrix(numNodesOut, numNodesIn);
            _weightVelocities = new DNAMatrix(numNodesOut, numNodesIn);

            biases = new DNAMatrix(numNodesOut, 1);
            _costGradientBias = new DNAMatrix(numNodesOut, 1);
            _biasVelocities = new DNAMatrix(numNodesOut, 1);

            InitializeRandomWeights(new System.Random());
        }

        public DNAMatrix CalculateOutputs (DNAMatrix inputs)
        {
            DNAMatrix outputs = (weights * inputs) + biases;

            for (int outputNode = 0; outputNode < outputs.Values.Length; outputNode ++)
            {
                outputs[outputNode] = activation.Activate(outputs.Values, outputNode);
            }

            return outputs;
        }

        public DNAMatrix CalculateOutputs(DNAMatrix inputs, DNALayerLearnData learnData)
        {
            learnData.inputs = inputs;

            //Calculate the outputs
            learnData.weightedInputs = (weights * inputs) + biases;

            //Apply Activation Function
            for (int outputNode = 0; outputNode < NumNodesOut; outputNode++)
            {
                learnData.activations[outputNode] = activation.Activate(learnData.weightedInputs.Values, outputNode);
            }
            return learnData.activations;
        }

        public void ApplyGradients(double learnRate, double regularization, double momentum)
        {
            double weightDecay = (1 - regularization * learnRate);

            //Calculate Velocities and Apply them to the respective matrices
            _weightVelocities = _weightVelocities * momentum - _costGradientWeight * learnRate;
            weights = weights * weightDecay + _weightVelocities;
            
            _biasVelocities = _biasVelocities * momentum - _costGradientBias * learnRate;
            biases += _biasVelocities;

            //Reset Gradients
            _costGradientWeight = new DNAMatrix(_costGradientWeight.Height, _costGradientWeight.Width);
            _costGradientBias = new DNAMatrix(_costGradientBias.Height, _costGradientBias.Width);
        }


        public void CalculateOutputLayerNodeValues(DNALayerLearnData layerLearnData, DNAMatrix expectedOutputs, ICost cost)
        {
            for (int i = 0; i < layerLearnData.nodeValues.Values.Length; i++)
            {
                double costDerivative = cost.CostDerivative(layerLearnData.activations[i], expectedOutputs[i]);
                double activationDerivative = activation.Derivative(layerLearnData.weightedInputs.Values, i);
                layerLearnData.nodeValues[i] = costDerivative * activationDerivative;
            }
        }

        //this one might be iffy
        public void CalculateHiddenLayerNodeValues(DNALayerLearnData layerLearnData, DNALayer oldLayer, DNAMatrix oldNodeValues)
        {

            DNAMatrix newNodeValues = oldNodeValues * oldLayer.weights.Transpose();

            for (int newNodeIndex = 0; newNodeIndex < newNodeValues.Values.Length; newNodeIndex++)
            {
                newNodeValues[newNodeIndex] *= activation.Derivative(layerLearnData.weightedInputs.Values, newNodeIndex);
            }

            layerLearnData.nodeValues = newNodeValues;

                /*
            for (int newNodeIndex = 0; newNodeIndex < numNodesOut; newNodeIndex++)
            {
                double newNodeValue = 0;
                for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.Length; oldNodeIndex++)
                {
                    //Get the partial derivative of weight with respect to input
                    double weightedInputDerivative = oldLayer.GetWeight(newNodeIndex, oldNodeIndex);
                    newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
                }
                newNodeValue *= activation.Derivative(layerLearnData.weightedInputs, newNodeIndex);
                layerLearnData.nodeValues[newNodeIndex] = newNodeValue;
            }
                */
        }
        public void UpdateGradients(DNALayerLearnData layerLearnData)
        {
            lock (_costGradientWeight)
            {

                _costGradientWeight += layerLearnData.nodeValues * layerLearnData.inputs.Transpose();

                /*
                for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
                {
                    double nodeValue = layerLearnData.nodeValues[nodeOut];
                    for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++)
                    {
                        // Evaluate the partial derivative: cost / weight of current connection
                        double derivativeCostWrtWeight = layerLearnData.inputs[nodeIn] * nodeValue;
                        // The costGradientW array stores these partial derivatives for each weight.
                        // Note: the derivative is being added to the array here because ultimately we want
                        // to calculate the average gradient across all the data in the training batch
                        costGradientWeight[GetFlatWeightIndex(nodeIn, nodeOut)] += derivativeCostWrtWeight;
                    }
                }
                */
            }

            // Update cost gradient with respect to biases (lock for multithreading)
            lock (_costGradientBias)
            {
                _costGradientBias += layerLearnData.nodeValues;

                /*
                for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++)
                {
                    // Evaluate partial derivative: cost / bias
                    double derivativeCostWrtBias = 1 * layerLearnData.nodeValues[nodeOut];
                    costGradientBias[nodeOut] += derivativeCostWrtBias;
                }
                */
            }
        }

        public void SetActivationFunction(IActivation activation)
        {
            this.activation = activation;
        }


        public void InitializeRandomWeights(System.Random rng)
        {
            for (int weightIndex = 0; weightIndex < weights.Values.Length; weightIndex++)
            {
                weights[weightIndex] = RandomInNormalDistribution(rng, 0, 1) / Mathf.Sqrt(NumNodesIn);
            }

            for (int biasIndex = 0; biasIndex < biases.Values.Length; biasIndex++)
            {
                biases[biasIndex] = RandomInNormalDistribution(rng, 0, 1) / Mathf.Sqrt(NumNodesIn);
            }

            double RandomInNormalDistribution(System.Random rng, double mean, double standardDeviation)
            {
                double x1 = 1 - rng.NextDouble();
                double x2 = 1 - rng.NextDouble();

                double y1 = Mathf.Sqrt(-2.0f * Mathf.Log((float)x1)) * Mathf.Cos(2.0f * Mathf.PI * (float)x2);
                return y1 * standardDeviation + mean;
            }
        }

    }
    
}

