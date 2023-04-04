using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using DNAMath;

namespace DNANeuralNet
{
    [System.Serializable]
    public class DNANeuralNetwork
    {
     
        public DNALayer[] layers;

        public int[] layerSizes;

        public ICost cost;
        System.Random rng;
        NetworkLearnData[] batchLearnData;

        public DNANeuralNetwork (DNANeuralNetworkInfo info)
        {
            //Initialize the neural networks

            layers = new DNALayer[info.layerInfos.Length];

            layers = createLayers(info);

            //Set the cost function
            SetCostFunction(Cost.GetCostFromType(info.costType));

        }
      
        DNALayer[] createLayers (DNANeuralNetworkInfo info)
        {
            List<DNALayer> layers = new List<DNALayer>();

            for (int i = 0; i < info.layerInfos.Length; i++)
            {
               

                DNALayer layer = null;
                LayerInfo layInfo = info.layerInfos[i];

                switch (info.layerInfos[i].type)
                {
                    case LayerTypes.Activation:

                        if (i - 1 >= 0)
                        {
                            layer = new ActivationLayer(layInfo.activation.activationType, layers[i - 1].outputSize);
                        } else
                        {
                            layer = new ActivationLayer(layInfo.activation.activationType, info.inputSize);
                        }
                           

                        break;
                    case LayerTypes.Filter:

                        if (i - 1 >= 0)
                        {
                            //Check if last layer has multiple outputs
                            layer = new FilterLayer(layInfo.filter.filterSize, layInfo.filter.numOfFilters, layInfo.filter.stride, layers[i - 1].outputSize, layers[i-1].outputMatNum);
                        } else
                        {
                            layer = new FilterLayer(layInfo.filter.filterSize, layInfo.filter.numOfFilters, layInfo.filter.stride, info.inputSize);
                        }
                        break;
                    case LayerTypes.Neural:

                        if (i - 1 >= 0)
                        {
                            if (i - 2 >= 0)
                            {
                               



                                layer = new NeuralLayer(layers[i - 1].iLayer.flattenLayer(layers[i-2].outputSize), layInfo.neural.outputSize);
                            } else
                            {
                                layer = new NeuralLayer(layers[i - 1].iLayer.flattenLayer(info.inputSize), layInfo.neural.outputSize);
                            }
                            
                        } else
                        {
                            layer = new NeuralLayer(info.inputSize.x * info.inputSize.y, layInfo.neural.outputSize);
                        }

                            

                        break;
                    case LayerTypes.Pooling:

                        if (i - 1 >= 0)
                        {
                            layer = new PoolingLayer(layInfo.pooling.size, layInfo.pooling.stride, layInfo.pooling.poolingType, layers[i - 1].outputSize, layers[i - 1].outputMatNum);
                        } else
                        {
                            layer = new PoolingLayer(layInfo.pooling.size, layInfo.pooling.stride, layInfo.pooling.poolingType, info.inputSize);
                        }

                        break;
                }

                layers.Add(layer);
            }

            return layers.ToArray();
        }

        public void SetCostFunction(ICost costFunction)
        {
            this.cost = costFunction;
        }

        public DNAMatrix[] CalculateOutputs(DNAMatrix input)
        {
            DNAMatrix[] inputs = new DNAMatrix[1];
            inputs[0] = input;

            foreach (DNALayer layer in layers)
            {
                inputs = layer.iLayer.CalculateOutputs(inputs);
            }

            return inputs;
        }
    }

    [System.Serializable]
    public class DNANeuralNetworkInfo
    {
        public Vector2Int inputSize;
        public LayerInfo[] layerInfos;
        public Cost.CostType costType;
    }

    [System.Serializable]
    public class LayerInfo
    {
        public LayerTypes type;

        public ActivationLayerInfo activation;
        public FilterLayerInfo filter;
        public NeuralLayerInfo neural;
        public PoolingLayerInfo pooling;
    }

    [System.Serializable]
    public enum LayerTypes
    {
        Activation,
        Filter,
        Neural, 
        Pooling,
    }

    [System.Serializable]
    public struct NeuralLayerInfo
    {
        public int outputSize;
    }

    [System.Serializable]
    public struct FilterLayerInfo 
    {
        //Determines size of the filter, (Change to int? Since the dimensions probably need to be square)
        public Vector2Int filterSize;
        public int numOfFilters;

        //Unless stride is always zero
        public int stride;
    }

    [System.Serializable]
    public struct ActivationLayerInfo
    {
        public Activation.ActivationType activationType;
    }

    [System.Serializable]
    public struct PoolingLayerInfo
    {
      

        public PoolingType poolingType;
        public Vector2Int size;
        public int stride;
    }

    public enum PoolingType
    {
        Max,
        Average,
        Min
    }

    public class DNALayerLearnData
    {
        //Theoretically only need inputs, weightedInputs? and nodeValues?
        public DNAMatrix[] inputs;
        public DNAMatrix[] weightedInputs;
        public DNAMatrix[] activations;
        public DNAMatrix[] outputs;

        //Node Values are for the derivatives
        public DNAMatrix[] nodeValues;

        //public double[] inputs;
       // public double[] weightedInputs;
       // public double[] activations;
       // public double[] nodeValues;

        public DNALayerLearnData(DNALayer layer)
        {
            //weightedInputs = new double[layer.numNodesOut];
            //activations = new double[layer.numNodesOut];
            //nodeValues = new double[layer.numNodesOut];




        }

    }

    public class DNANetworkLearnData
    {
        public DNALayerLearnData[] layerData;

        public DNANetworkLearnData(DNALayer[] layers)
        {
            layerData = new DNALayerLearnData[layers.Length];
            for (int i = 0; i < layers.Length; i++)
            {
                layerData[i] = new DNALayerLearnData(layers[i]);
            }
        }
    }

}