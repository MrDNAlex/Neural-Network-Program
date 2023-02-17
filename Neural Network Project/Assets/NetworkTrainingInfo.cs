using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class NetworkTrainingInfo 
{

    public int[] neuralNetSize;



    public NetworkTrainingInfo (int[] size)
    {
        this.neuralNetSize = size;
 

    }

}
