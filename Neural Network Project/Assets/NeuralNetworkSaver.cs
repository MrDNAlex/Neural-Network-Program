using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

[System.Serializable]
public class NeuralNetworkSaver
{

    public LayerSaver[] layers;

    public double inputVal;


    public NeuralNetworkSaver (NeuralNetwork neuro)
    {
        layers = new LayerSaver[neuro.layers.Length];

        for (int i = 0; i < neuro.layers.Length; i ++)
        {
            layers[i] = new LayerSaver(neuro.layers[i]);
        }

    }

    public NeuralNetwork createNetwork (NeuralNetworkSaver saver)
    {
        /*
        List<int> ints = new List<int>();

        ints.Add(saver.layers[0].numNodesIn);
        ints.Add(saver.layers[0].numNodesOut);

        for (int i = 1; i < saver.layers.Length; i ++)
        {
            ints.Add(saver.layers[i].numNodesOut);
        }
        */

        NeuralNetwork neuro = new NeuralNetwork();

        neuro.layers = new Layer[saver.layers.Length];

        for (int i = 0; i < saver.layers.Length; i++)
        {
            neuro.layers[i] = new Layer(saver.layers[i]);
        }

        return neuro;

    }


}
