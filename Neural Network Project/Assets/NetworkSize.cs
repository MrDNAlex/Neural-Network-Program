using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class NetworkSize
{

    public int[] neuralNetSize;

    public NetworkSize(int[] size)
    {
        this.neuralNetSize = size;


    }
}
