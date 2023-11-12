using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using DNAMath;
using System.Threading.Tasks;

namespace DNANeuralNet
{
    [System.Serializable]
    public class DNALayer
    {
        [SerializeField]
        private int _numNodeIn;

        [SerializeField]
        private int _numNodeOut;

        public int NumNodesIn { get { return _numNodeIn; } set { _numNodeIn = value; } }

        public int NumNodesOut { get { return _numNodeOut; } set { _numNodeOut = value; } }

        public DNAMatrix weights;
        public DNAMatrix biases;

        //Cost Gradient With respect to weight and biases
        private DNAMatrix _costGradientWeight;
        private DNAMatrix _costGradientBias;

        //Momentum
        private DNAMatrix _weightVelocities;
        private DNAMatrix _biasVelocities;

        [SerializeField]
        public IDNAActivation activation;

        //Compute Buffers
        ComputeBuffer weightsVals;
        ComputeBuffer biasVals;

        ComputeBuffer weightsDim = new ComputeBuffer(1, sizeof(uint) * 2);
        ComputeBuffer biasDim = new ComputeBuffer(1, sizeof(uint) * 2);

        public DNALayer(int numNodesIn, int numNodesOut)
        {
            this.NumNodesIn = numNodesIn;
            this.NumNodesOut = numNodesOut;
            activation = new DNAActivation.Sigmoid();

            weights = new DNAMatrix(numNodesOut, numNodesIn);
            _costGradientWeight = new DNAMatrix(numNodesOut, numNodesIn);
            _weightVelocities = new DNAMatrix(numNodesOut, numNodesIn);

            biases = new DNAMatrix(numNodesOut, 1);
            _costGradientBias = new DNAMatrix(numNodesOut, 1);
            _biasVelocities = new DNAMatrix(numNodesOut, 1);

            InitializeRandomWeights(new System.Random());

            weightsVals = new ComputeBuffer(weights.Length, sizeof(double));
            biasVals = new ComputeBuffer(biases.Length, sizeof(double));

            UpdateComputeBuffers();
        }

        public DNAMatrix CalculateOutputs(DNAMatrix inputs)
        {
            if (layerOutputGPU != null)
                return layerOutputCalculationGPU(inputs);
            else
                return activation.Activate((weights * inputs) + biases);
        }

        public DNAMatrix CalculateOutputs(DNAMatrix inputs, DNALayerLearnData learnData)
        {
            learnData.inputs = inputs;

            if (layerOutputGPU != null)
            {
                (DNAMatrix weightedInputs, DNAMatrix activation) = layerOutputCalculationTrainingGPU(inputs);

                //Calculate the outputs
                learnData.weightedInputs = weightedInputs;

                //Apply Activation Function
                learnData.activations = activation;
            }
            else
            {
                //Calculate the outputs
                learnData.weightedInputs = (weights * inputs) + biases;

                //Apply Activation Function
                learnData.activations = activation.Activate(learnData.weightedInputs);
            }

            return learnData.activations;
        }

        public DNAMatrix[] ParallelCalculateOutputs(DNAMatrix[] inputs, DNALayerLearnData[] learnData)
        {
            (DNAMatrix[] weightedInputs, DNAMatrix[] activation) = parallelLayerOutputCalculationTrainingGPU(inputs);

            for (int i = 0; i < inputs.Length; i++)
            {
                learnData[i].inputs = inputs[i];

                //Calculate the outputs
                learnData[i].weightedInputs = weightedInputs[i];

                //Apply Activation Function
                learnData[i].activations = activation[i];
            }

            return activation;
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

            UpdateComputeBuffers();
        }

        public void CalculateOutputLayerNodeValues(DNALayerLearnData layerLearnData, DNAMatrix expectedOutputs, IDNACost cost)
        {
            DNAMatrix costDerivative = cost.CostDerivative(layerLearnData.activations, expectedOutputs);
            DNAMatrix activationDerivative = activation.Derivative(layerLearnData.weightedInputs);

            for (int i = 0; i < layerLearnData.nodeValues.Values.Length; i++)
                layerLearnData.nodeValues[i] = costDerivative[i] * activationDerivative[i];
        }

        public void CalculateHiddenLayerNodeValues(DNALayerLearnData layerLearnData, DNALayer oldLayer, DNAMatrix oldNodeValues)
        {
            DNAMatrix newNodeValues = oldLayer.weights.Transpose() * oldNodeValues;

            DNAMatrix derivative = activation.Derivative(layerLearnData.weightedInputs);

            for (int newNodeIndex = 0; newNodeIndex < newNodeValues.Values.Length; newNodeIndex++)
                newNodeValues[newNodeIndex] *= derivative[newNodeIndex];

            layerLearnData.nodeValues = newNodeValues;
        }

        public void UpdateGradients(DNALayerLearnData layerLearnData)
        {
            lock (_costGradientWeight)
            {
                _costGradientWeight += layerLearnData.nodeValues * layerLearnData.inputs.Transpose();
            }

            // Update cost gradient with respect to biases (lock for multithreading)
            lock (_costGradientBias)
            {
                _costGradientBias += layerLearnData.nodeValues;
            }
        }

        public void SetActivationFunction(IDNAActivation activation)
        {
            this.activation = activation;
        }

        public void InitializeRandomWeights(System.Random rng)
        {
            for (int weightIndex = 0; weightIndex < weights.Values.Length; weightIndex++)
            {
                weights[weightIndex] = RandomInNormalDistribution(rng, 0, 1) / Mathf.Sqrt(NumNodesIn);
            }

            double RandomInNormalDistribution(System.Random rng, double mean, double standardDeviation)
            {
                double x1 = 1 - rng.NextDouble();
                double x2 = 1 - rng.NextDouble();

                double y1 = Mathf.Sqrt(-2.0f * Mathf.Log((float)x1)) * Mathf.Cos(2.0f * Mathf.PI * (float)x2);
                return y1 * standardDeviation + mean;
            }
        }

        private void UpdateComputeBuffers()
        {
            weightsDim.SetData(new uint[] { (uint)weights.Width, (uint)weights.Height });
            biasDim.SetData(new uint[] { (uint)biases.Width, (uint)biases.Height });

            weightsVals.SetData(weights.Values);
            biasVals.SetData(biases.Values);
        }

        public static ComputeShader layerOutputGPU;

        public static ComputeShader parallelLayerOutputGPU;

        // public static ComputeShader parallelLoader;

        public static ComputeShader parallelOutputLayer;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterAssembliesLoaded)]
        public static void loadAssets()
        {
            layerOutputGPU = Resources.Load<ComputeShader>("LayerOutputCalculation");
            parallelLayerOutputGPU = Resources.Load<ComputeShader>("ParrallelLayerOperation");
            parallelOutputLayer = Resources.Load<ComputeShader>("ParallelOutputLayer");

            if (layerOutputGPU != null)
            {
                Debug.Log("Loaded!");
            }
            if (parallelLayerOutputGPU != null)
            {
                Debug.Log("Parallel Loaded!");
            }
            if (parallelOutputLayer != null)
            {
                Debug.Log("Parallel Output Loaded!");
            }
        }

        private DNAMatrix layerOutputCalculationGPU(DNAMatrix inputs)
        {
            DNAMatrix activation = new DNAMatrix(0, 0);
            if (weights.Width == inputs.Height)
            {
                activation = new DNAMatrix(weights.Height, inputs.Width);

                ComputeShader computeShader = layerOutputGPU;

                //Setup Compute Buffers
                ComputeBuffer inputsVals = new ComputeBuffer(inputs.Length, sizeof(double));

                ComputeBuffer inputsDim = new ComputeBuffer(1, sizeof(uint) * 2);

                ComputeBuffer activationVals = new ComputeBuffer(activation.Length, sizeof(double));
                ComputeBuffer weightedInputVals = new ComputeBuffer(activation.Length, sizeof(double));

                ComputeBuffer dimensions = new ComputeBuffer(3, sizeof(uint) * 2);

                ComputeBuffer activationFunction = new ComputeBuffer(1, sizeof(int));

                //Set Data
                inputsVals.SetData(inputs.Values);

                inputsDim.SetData(new uint[] { (uint)inputs.Width, (uint)inputs.Height });

                activationFunction.SetData(new int[] { this.activation.GetActivationFunctionIndex() });

                dimensions.SetData(new uint[] { (uint)weights.Width, (uint)weights.Height, (uint)biases.Width, (uint)biases.Height, (uint)inputs.Width, (uint)inputs.Height });

                //Set Buffers
                computeShader.SetBuffer(0, "weights", weightsVals);
                computeShader.SetBuffer(0, "inputs", inputsVals);
                computeShader.SetBuffer(0, "bias", biasVals);

                computeShader.SetBuffer(0, "dimensions", dimensions);

                computeShader.SetBuffer(0, "weightedInputs", weightedInputVals);
                computeShader.SetBuffer(0, "activation", activationVals);

                computeShader.SetBuffer(0, "activationFunction", activationFunction);

                //Calculate
                computeShader.Dispatch(0, activation.Width, activation.Height, 1);

                //Receaive Result
                activationVals.GetData(activation.Values);

                inputsVals.Release();
                activationVals.Release();
                weightedInputVals.Release();

                inputsDim.Release();
                activationFunction.Release();
                dimensions.Release();
            }
            else
                Debug.Log("Error, Dimensions don't match");

            return activation;
        }

        private (DNAMatrix weightedInputs, DNAMatrix activation) layerOutputCalculationTrainingGPU(DNAMatrix inputs)
        {
            DNAMatrix activation = new DNAMatrix(0, 0);
            DNAMatrix weightedInputs = new DNAMatrix(0, 0);
            if (weights.Width == inputs.Height)
            {
                activation = new DNAMatrix(weights.Height, inputs.Width);
                weightedInputs = new DNAMatrix(weights.Height, inputs.Width);

                ComputeShader computeShader = layerOutputGPU;

                //Setup Compute Buffers
                ComputeBuffer inputsVals = new ComputeBuffer(inputs.Length, sizeof(double));

                ComputeBuffer inputsDim = new ComputeBuffer(1, sizeof(uint) * 2);

                ComputeBuffer activationVals = new ComputeBuffer(activation.Length, sizeof(double));
                ComputeBuffer weightedInputVals = new ComputeBuffer(activation.Length, sizeof(double));

                //Set Data
                inputsVals.SetData(inputs.Values);

                weightsDim.SetData(new uint[] { (uint)weights.Width, (uint)weights.Height });
                inputsDim.SetData(new uint[] { (uint)inputs.Width, (uint)inputs.Height });
                biasDim.SetData(new uint[] { (uint)biases.Width, (uint)biases.Height });

                ComputeBuffer activationFunction = new ComputeBuffer(1, sizeof(int));
                activationFunction.SetData(new int[] { this.activation.GetActivationFunctionIndex() });

                computeShader.SetBuffer(0, "activationFunction", activationFunction);

                //Set Buffers
                computeShader.SetBuffer(0, "weights", weightsVals);
                computeShader.SetBuffer(0, "inputs", inputsVals);
                computeShader.SetBuffer(0, "bias", biasVals);

                computeShader.SetBuffer(0, "weightsDim", weightsDim);
                computeShader.SetBuffer(0, "inputsDim", inputsDim);
                computeShader.SetBuffer(0, "biasDim", biasDim);

                computeShader.SetBuffer(0, "weightedInputs", weightedInputVals);
                computeShader.SetBuffer(0, "activation", activationVals);

                //Calculate
                computeShader.Dispatch(0, activation.Width, activation.Height, 1);

                //Receaive Result
                activationVals.GetData(activation.Values);
                weightedInputVals.GetData(weightedInputs.Values);

                inputsVals.Release();
                activationVals.Release();
                weightedInputVals.Release();

                inputsDim.Release();
                activationFunction.Release();
            }
            else
                Debug.Log("Error, Dimensions don't match");

            return (weightedInputs, activation);
        }


        private (DNAMatrix[] weightedInputs, DNAMatrix[] activation) parallelLayerOutputCalculationTrainingGPU(DNAMatrix[] inputs)
        {
            DNAMatrix[] activation = new DNAMatrix[inputs.Length];
            DNAMatrix[] weightedInputs = new DNAMatrix[inputs.Length];
            if (weights.Width == inputs[0].Height)
            {
                int inputsLength = inputs.Length * inputs[0].Length;
                int outputsLength = inputs.Length * weights.Height * inputs[0].Width;

                for (int i = 0; i < inputs.Length; i++)
                {
                    activation[i] = new DNAMatrix(weights.Height, inputs[0].Width);
                    weightedInputs[i] = new DNAMatrix(weights.Height, inputs[0].Width);
                }

                ComputeShader computeShader = parallelLayerOutputGPU;

                //Setup Compute Buffers
                ComputeBuffer dimensions = new ComputeBuffer(3, sizeof(uint) * 2);

                ComputeBuffer inputsVals = new ComputeBuffer(inputsLength, sizeof(double));

                ComputeBuffer activationVals = new ComputeBuffer(outputsLength, sizeof(double));
                ComputeBuffer weightedInputVals = new ComputeBuffer(outputsLength, sizeof(double));

                ComputeBuffer activationFunction = new ComputeBuffer(1, sizeof(int));

                //Set Data
                inputsVals.SetData(GetInputArray(inputs));

                dimensions.SetData(new uint[] { (uint)weights.Width, (uint)weights.Height, (uint)biases.Width, (uint)biases.Height, (uint)inputs[0].Width, (uint)inputs[0].Height });

                activationFunction.SetData(new int[] { this.activation.GetActivationFunctionIndex() });

                //Set Buffers
                computeShader.SetBuffer(0, "weights", weightsVals);
                computeShader.SetBuffer(0, "inputs", inputsVals);
                computeShader.SetBuffer(0, "bias", biasVals);

                computeShader.SetBuffer(0, "dimensions", dimensions);

                computeShader.SetBuffer(0, "weightedInputs", weightedInputVals);
                computeShader.SetBuffer(0, "activation", activationVals);

                computeShader.SetBuffer(0, "activationFunction", activationFunction);

                //Calculate
                computeShader.Dispatch(0, activation[0].Width, activation[0].Height, inputs.Length);

                double[] activationsOutput = new double[inputs.Length * activation[0].Length];
                double[] weightedInputOutput = new double[inputs.Length * weightedInputs[0].Length];

                //Receive Result
                activationVals.GetData(activationsOutput);
                weightedInputVals.GetData(weightedInputOutput);

                //Format Correctly
                ConvertToMatrices(weightedInputs, weightedInputOutput, activation, activationsOutput);

                //Clear Memory
                inputsVals.Release();
                activationVals.Release();
                weightedInputVals.Release();

                dimensions.Release();
                activationFunction.Release();
            }
            else
                Debug.Log("Error, Dimensions don't match");

            return (weightedInputs, activation);
        }

        private double[] GetInputArray(DNAMatrix[] inputs)
        {
            double[] inputValues = new double[inputs.Length * inputs[0].Length];

            int count = 0;
            foreach (DNAMatrix matrix in inputs)
            {
                Array.Copy(matrix.Values, 0, inputValues, count, matrix.Values.Length);
                count += matrix.Values.Length;
            }

            return inputValues;
        }

        private void ConvertToMatrices(DNAMatrix[] weightedInputs, double[] weightedInputsOutput, DNAMatrix[] activations, double[] activationsOutput)
        {
            for (int i = 0; i < weightedInputs.Length; i++)
            {
                Array.Copy(weightedInputsOutput, i * weightedInputs[i].Length, weightedInputs[i].Values, 0, weightedInputs[i].Length);
                Array.Copy(activationsOutput, i * activations[i].Length, activations[i].Values, 0, activations[i].Length);
            }
        }

        //Parallel Version
        public void ParallelCalculateOutputLayerNodeValues(DNALayerLearnData[] layerLearnData, DNAMatrix[] expectedOutput, IDNACost cost)
        {
            int expectedOutputLength = expectedOutput.Length *  expectedOutput[0].Length;
            int weightedInputLength = layerLearnData.Length * layerLearnData[0].weightedInputs.Length;
            int activationLength = layerLearnData.Length * layerLearnData[0].activations.Length;
            int nodeValuesLength = layerLearnData.Length * layerLearnData[0].nodeValues.Length;

            ComputeShader computeShader = parallelOutputLayer;

            //Setup Compute Buffers
            ComputeBuffer dimensions = new ComputeBuffer(1, sizeof(uint) * 2);

            ComputeBuffer weightedInputs = new ComputeBuffer(weightedInputLength, sizeof(double));

            ComputeBuffer activations = new ComputeBuffer(activationLength, sizeof(double));

            ComputeBuffer expectedOutputs = new ComputeBuffer(expectedOutputLength, sizeof(double));

            ComputeBuffer nodeValues = new ComputeBuffer(nodeValuesLength, sizeof(double));

            ComputeBuffer derivativeTypes = new ComputeBuffer(2, sizeof(int));

            //Set Data
            dimensions.SetData(new uint[] { (uint)expectedOutput[0].Width, (uint)expectedOutput[0].Height });

            weightedInputs.SetData(GetWeightedInputs(layerLearnData));

            activations.SetData(GetActivationInputs(layerLearnData));

            expectedOutputs.SetData(GetInputArray(expectedOutput));

            derivativeTypes.SetData(new int[] { cost.GetCostIndex(), this.activation.GetActivationFunctionIndex() });

            //Set Buffers
            computeShader.SetBuffer(0, "dimensions", dimensions);

            computeShader.SetBuffer(0, "weightedInputs", weightedInputs);
            computeShader.SetBuffer(0, "activations", activations);
            computeShader.SetBuffer(0, "expectedOutputs", expectedOutputs);
            computeShader.SetBuffer(0, "nodeValues", nodeValues);

            computeShader.SetBuffer(0, "derivativeType", derivativeTypes);

            //Calculate
            computeShader.Dispatch(0, expectedOutput[0].Width, expectedOutput[0].Height, expectedOutput.Length);

            double[] nodeVals = new double[nodeValuesLength];

            //Receive Result
            nodeValues.GetData(nodeVals);

            //Format Correctly
            SetNodeValues(nodeVals, layerLearnData);

            //Clear Memory
            weightedInputs.Release();
            activations.Release();
            expectedOutputs.Release();
            nodeValues.Release();

            dimensions.Release();
            derivativeTypes.Release();
        }

        private double[] GetWeightedInputs (DNALayerLearnData[] layerLearnData)
        {
            double[] weightedInputs = new double[layerLearnData.Length * layerLearnData[0].weightedInputs.Length];

            int count = 0;
            foreach (DNALayerLearnData layer in layerLearnData)
            {
                Array.Copy(layer.weightedInputs.Values, 0, weightedInputs, count, layer.weightedInputs.Length);
                count += layer.weightedInputs.Values.Length;
            }

            return weightedInputs;
        }

        private double[] GetActivationInputs(DNALayerLearnData[] layerLearnData)
        {
            double[] activations = new double[layerLearnData.Length * layerLearnData[0].activations.Length];

            int count = 0;
            foreach (DNALayerLearnData layer in layerLearnData)
            {
                Array.Copy(layer.activations.Values, 0, activations, count, layer.activations.Length);
                count += layer.activations.Values.Length;
            }

            return activations;
        }

        private void SetNodeValues (double[] nodeValues, DNALayerLearnData[] layerLearnData)
        {
            for (int i = 0; i < layerLearnData.Length; i++)
            {
                Array.Copy(nodeValues, i * layerLearnData[i].nodeValues.Length, layerLearnData[i].nodeValues.Values, 0, layerLearnData[i].nodeValues.Length);
            }
        }
    }
}