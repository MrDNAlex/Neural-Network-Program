using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class NetworkTrainingInfo 
{
    //Add extra info like calculated cost before and after, accuracy, reference to neural network and more
    public int[] neuralNetSize;
    public float learnRate;
    public int dataPerBatch;

    public NetworkTrainingInfo (int[] size, float learnRate, int dataPerBatch)
    {
        this.neuralNetSize = size;
        this.learnRate = learnRate;
        this.dataPerBatch = dataPerBatch;
 

    }

}
