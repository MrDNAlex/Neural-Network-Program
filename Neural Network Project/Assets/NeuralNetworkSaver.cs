using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using DNANeuralNetwork;


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

    /*
    public NeuralNetwork createNetwork (NeuralNetworkSaver saver)
    {
       

        NeuralNetwork neuro = new NeuralNetwork();

        neuro.layers = new Layer[saver.layers.Length];

        for (int i = 0; i < saver.layers.Length; i++)
        {
            neuro.layers[i] = new Layer(saver.layers[i]);
        }

        return neuro;

    }
    */


}

