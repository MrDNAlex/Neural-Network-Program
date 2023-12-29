using DNAMath;
using System;
using System.IO;
using UnityEngine;

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

        public void ParallelCalculateHiddenLayerNodeValues(DNALayerLearnData[] layerLearnData, DNALayer oldLayer, DNAMatrix[] oldNodeValues)
        {
            ParallelHiddenLayerNodeCalc(layerLearnData, oldLayer, oldNodeValues);
        }

        public void UpdateGradients(DNALayerLearnData layerLearnData)
        {
            //Lock for Parallel Processing
            lock (_costGradientWeight)
            {
                _costGradientWeight += layerLearnData.nodeValues * layerLearnData.inputs.Transpose();
            }

            lock (_costGradientBias)
            {
                _costGradientBias += layerLearnData.nodeValues;
            }
        }

        public void ParallelUpdateGradients(DNALayerLearnData[] layerLearnData)
        {
            _costGradientBias += ParallelUpdateGradientsBias(layerLearnData);

            _costGradientWeight += ParallelUpdateGradientsWeights(layerLearnData);
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

        public static ComputeShader parallelUpdateGradientsWeights;
        public static ComputeShader parallelUpdateGradientsBias;

        public static ComputeShader UpdateGradientsWeights;

        public static ComputeShader UpdateGradientBias;

        public static ComputeShader ParallelHiddenLayerNode;
        public static ComputeShader HiddenLayerNode;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterAssembliesLoaded)]
        public static void loadAssets()
        {
            layerOutputGPU = Resources.Load<ComputeShader>("LayerOutputCalculation");
            parallelLayerOutputGPU = Resources.Load<ComputeShader>("ParrallelLayerOperation");
            parallelOutputLayer = Resources.Load<ComputeShader>("ParallelOutputLayer");
            parallelUpdateGradientsWeights = Resources.Load<ComputeShader>("ParallelUpdateGradientsWeights");
            parallelUpdateGradientsBias = Resources.Load<ComputeShader>("ParallelUpdateGradientsBias");
            UpdateGradientsWeights = Resources.Load<ComputeShader>("UpdateGradientWeights");
            ParallelHiddenLayerNode = Resources.Load<ComputeShader>("ParallelHiddenLayerNode");
            HiddenLayerNode = Resources.Load<ComputeShader>("HiddenLayerNode");
            UpdateGradientBias = Resources.Load<ComputeShader>("UpdateGradientBias");

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
            if (parallelUpdateGradientsWeights != null)
            {
                Debug.Log("Parallel Gradients Weights Loaded!");
            }
            if (parallelUpdateGradientsBias != null)
            {
                Debug.Log("Parallel Gradients Bias Loaded!");
            }
            if (UpdateGradientsWeights)
            {
                Debug.Log("Update Weights Loaded!");
            }
            if (ParallelHiddenLayerNode != null)
            {
                Debug.Log("Parallel Hidden Layer Node");
            }

            if (HiddenLayerNode != null)
            {
                Debug.Log("Hidden Layer Node");
            }

            if (UpdateGradientBias != null)
            {
                Debug.Log("Update Bias Loaded");
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

                ComputeBuffer activationVals = new ComputeBuffer(activation.Length, sizeof(double));
                ComputeBuffer weightedInputVals = new ComputeBuffer(activation.Length, sizeof(double));

                ComputeBuffer dimensions = new ComputeBuffer(3, sizeof(int) * 2);

                ComputeBuffer activationFunction = new ComputeBuffer(1, sizeof(int));

                //Set Data
                inputsVals.SetData(inputs.Values);

                activationFunction.SetData(new int[] { this.activation.GetActivationFunctionIndex() });

                dimensions.SetData(new int[] { weights.Width, weights.Height, biases.Width, biases.Height, inputs.Width, inputs.Height });

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

                ComputeBuffer inputsVals = new ComputeBuffer(inputs.Length, sizeof(double));

                ComputeBuffer activationVals = new ComputeBuffer(activation.Length, sizeof(double));
                ComputeBuffer weightedInputVals = new ComputeBuffer(activation.Length, sizeof(double));

                ComputeBuffer dimensions = new ComputeBuffer(3, sizeof(int) * 2);

                ComputeBuffer activationFunction = new ComputeBuffer(1, sizeof(int));

                //Set Data
                inputsVals.SetData(inputs.Values);

                activationFunction.SetData(new int[] { this.activation.GetActivationFunctionIndex() });

                dimensions.SetData(new int[] { weights.Width, weights.Height, biases.Width, biases.Height, inputs.Width, inputs.Height });

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
                weightedInputVals.GetData(weightedInputs.Values);

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
                ComputeBuffer dimensions = new ComputeBuffer(3, sizeof(int) * 2);
                ComputeBuffer inputsVals = new ComputeBuffer(inputsLength, sizeof(double));
                ComputeBuffer activationVals = new ComputeBuffer(outputsLength, sizeof(double));
                ComputeBuffer weightedInputVals = new ComputeBuffer(outputsLength, sizeof(double));
                ComputeBuffer activationFunction = new ComputeBuffer(1, sizeof(int));

                //Set Data
                inputsVals.SetData(GetInputArray(inputs));

                dimensions.SetData(new int[] { weights.Width, weights.Height, biases.Width, biases.Height, inputs[0].Width, inputs[0].Height });

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
            int expectedOutputLength = expectedOutput.Length * expectedOutput[0].Length;
            int weightedInputLength = layerLearnData.Length * layerLearnData[0].weightedInputs.Length;
            int activationLength = layerLearnData.Length * layerLearnData[0].activations.Length;
            int nodeValuesLength = layerLearnData.Length * layerLearnData[0].nodeValues.Length;

            ComputeShader computeShader = parallelOutputLayer;

            //Setup Compute Buffers
            ComputeBuffer dimensions = new ComputeBuffer(1, sizeof(int) * 2);

            ComputeBuffer weightedInputs = new ComputeBuffer(weightedInputLength, sizeof(double));

            ComputeBuffer activations = new ComputeBuffer(activationLength, sizeof(double));

            ComputeBuffer expectedOutputs = new ComputeBuffer(expectedOutputLength, sizeof(double));

            ComputeBuffer nodeValues = new ComputeBuffer(nodeValuesLength, sizeof(double));

            ComputeBuffer derivativeTypes = new ComputeBuffer(2, sizeof(int));

            //Set Data
            dimensions.SetData(new int[] { expectedOutput[0].Width, expectedOutput[0].Height });

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

        private double[] GetWeightedInputs(DNALayerLearnData[] layerLearnData)
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

        private void SetNodeValues(double[] nodeValues, DNALayerLearnData[] layerLearnData)
        {
            for (int i = 0; i < layerLearnData.Length; i++)
            {
                Array.Copy(nodeValues, i * layerLearnData[i].nodeValues.Length, layerLearnData[i].nodeValues.Values, 0, layerLearnData[i].nodeValues.Length);
            }
        }

        public DNAMatrix ParallelUpdateGradientsWeights(DNALayerLearnData[] layerLearnData)
        {
            int inputsLength = layerLearnData.Length * layerLearnData[0].inputs.Length;
            int nodeValuesLength = layerLearnData.Length * layerLearnData[0].nodeValues.Length;
            int costGradientWeightLength = _costGradientWeight.Length;

            ComputeShader computeShader = parallelUpdateGradientsWeights;

            //Setup Compute Buffers
            ComputeBuffer dimensions = new ComputeBuffer(3, sizeof(int) * 2);

            ComputeBuffer inputsValues = new ComputeBuffer(inputsLength, sizeof(double));

            ComputeBuffer nodeValuesValues = new ComputeBuffer(nodeValuesLength, sizeof(double));

            ComputeBuffer weightsValues = new ComputeBuffer(costGradientWeightLength, sizeof(double));

            //Set Data
            dimensions.SetData(new int[] { layerLearnData[0].nodeValues.Width, layerLearnData[0].nodeValues.Height, layerLearnData[0].inputs.Height, layerLearnData[0].inputs.Width, _costGradientWeight.Width, _costGradientWeight.Height });

            inputsValues.SetData(GetGradientInputVals(layerLearnData));

            nodeValuesValues.SetData(GetGradientNodeVals(layerLearnData));

            weightsValues.SetData(new double[costGradientWeightLength]);

            //Set Buffers
            computeShader.SetBuffer(0, "dimensions", dimensions);

            computeShader.SetBuffer(0, "inputs", inputsValues);
            computeShader.SetBuffer(0, "nodeValues", nodeValuesValues);
            computeShader.SetBuffer(0, "costGradientWeight", weightsValues);

            //Calculate
            computeShader.Dispatch(0, _costGradientWeight.Width, _costGradientWeight.Height, 1); //layerLearnData.Length

            //double[] costGradientWeight = new double[costGradientWeightLength];
            DNAMatrix costGradientWeight = new DNAMatrix(_costGradientWeight.Height, _costGradientWeight.Width);

            //Receive Result
            weightsValues.GetData(costGradientWeight.Values);

            //Set Values
            // _costGradientWeight.Values = costGradientWeight;

            //Clear Memory
            inputsValues.Release();
            nodeValuesValues.Release();
            weightsValues.Release();

            dimensions.Release();

            return costGradientWeight;
        }

        public DNAMatrix ParallelUpdateGradientsBias(DNALayerLearnData[] layerLearnData)
        {
            int nodeValuesLength = layerLearnData.Length * layerLearnData[0].nodeValues.Length;
            int costGradientBiasLength = _costGradientBias.Length;

            ComputeShader computeShader = parallelUpdateGradientsBias;

            //Setup Compute Buffers
            ComputeBuffer dimensions = new ComputeBuffer(1, sizeof(int) * 2);

            ComputeBuffer nodeValuesValues = new ComputeBuffer(nodeValuesLength, sizeof(double));

            ComputeBuffer biasValues = new ComputeBuffer(costGradientBiasLength, sizeof(double));

            //Set Data
            dimensions.SetData(new int[] { layerLearnData[0].nodeValues.Width, layerLearnData[0].nodeValues.Height });

            nodeValuesValues.SetData(GetGradientNodeVals(layerLearnData));

            biasValues.SetData(new double[costGradientBiasLength]);

            //Set Buffers
            computeShader.SetBuffer(0, "dimensions", dimensions);

            computeShader.SetBuffer(0, "nodeValues", nodeValuesValues);
            computeShader.SetBuffer(0, "costGradientBias", biasValues);

            //Calculate
            computeShader.Dispatch(0, _costGradientBias.Width, _costGradientBias.Height, 1); //layerLearnData.Length

            DNAMatrix costGradientBias = new DNAMatrix(_costGradientBias.Height, _costGradientBias.Width);

            //Receive Result
            biasValues.GetData(costGradientBias.Values);

            //Clear Memory
            nodeValuesValues.Release();
            biasValues.Release();
            dimensions.Release();

            return costGradientBias;
        }

        private double[] GetGradientInputVals(DNALayerLearnData[] layerLearnData)
        {
            double[] inputs = new double[layerLearnData.Length * layerLearnData[0].inputs.Length];

            int count = 0;
            foreach (DNALayerLearnData layer in layerLearnData)
            {
                Array.Copy(layer.inputs.Values, 0, inputs, count, layer.inputs.Length); //Inputs are normally transposed
                count += layer.inputs.Values.Length;
            }

            return inputs;
        }

        private double[] GetGradientNodeVals(DNALayerLearnData[] layerLearnData)
        {
            double[] nodeVals = new double[layerLearnData.Length * layerLearnData[0].nodeValues.Length];

            string values = "";
            int count = 0;
            foreach (DNALayerLearnData layer in layerLearnData)
            {
                Array.Copy(layer.nodeValues.Values, 0, nodeVals, count, layer.nodeValues.Length);
                count += layer.nodeValues.Values.Length;
                values += "\n";
            }

            return nodeVals;
        }

        public DNAMatrix GetUpdateGradientBias(DNALayerLearnData layerLearnData)
        {
            int nodeValuesLength = layerLearnData.nodeValues.Length;
            int costGradientBiasLength = _costGradientBias.Length;

            ComputeShader computeShader = UpdateGradientBias;

            //Setup Compute Buffers
            ComputeBuffer dimensions = new ComputeBuffer(1, sizeof(int) * 2);

            ComputeBuffer nodeValuesValues = new ComputeBuffer(nodeValuesLength, sizeof(double));

            ComputeBuffer biasValues = new ComputeBuffer(costGradientBiasLength, sizeof(double));

            //Set Data
            dimensions.SetData(new int[] { layerLearnData.nodeValues.Width, layerLearnData.nodeValues.Height });

            nodeValuesValues.SetData(layerLearnData.nodeValues.Values);

            biasValues.SetData(new double[costGradientBiasLength]);

            //Set Buffers
            computeShader.SetBuffer(0, "dimensions", dimensions);

            computeShader.SetBuffer(0, "nodeValues", nodeValuesValues);
            computeShader.SetBuffer(0, "costGradientBias", biasValues);

            //Calculate
            Debug.Log($"Width: {_costGradientBias.Width}  Height:{_costGradientBias.Height}");
            computeShader.Dispatch(0, _costGradientBias.Width, _costGradientBias.Height, 1);

            DNAMatrix costGradientBias = new DNAMatrix(_costGradientBias.Height, _costGradientBias.Width);

            //Receive Result
            biasValues.GetData(costGradientBias.Values);

            costGradientBias.DisplayMat();
            //Set Values
            // _costGradientBias.Values = costGradientBias;

            //Clear Memory
            nodeValuesValues.Release();
            biasValues.Release();
            dimensions.Release();

            return costGradientBias;
        }

        public DNAMatrix UpdateGradientsWeight(DNALayerLearnData layerLearnData)
        {
            int inputsLength = layerLearnData.inputs.Length;
            int nodeValuesLength = layerLearnData.nodeValues.Length;
            int costGradientWeightLength = _costGradientWeight.Length;

            ComputeShader computeShader = UpdateGradientsWeights;

            //Setup Compute Buffers
            ComputeBuffer dimensions = new ComputeBuffer(2, sizeof(uint) * 2);

            ComputeBuffer nodeValuesValues = new ComputeBuffer(nodeValuesLength, sizeof(double));

            ComputeBuffer inputsValues = new ComputeBuffer(inputsLength, sizeof(double));

            ComputeBuffer weightsValues = new ComputeBuffer(costGradientWeightLength, sizeof(double));

            //Set Data
            dimensions.SetData(new uint[] { (uint)layerLearnData.nodeValues.Width, (uint)layerLearnData.nodeValues.Height, (uint)layerLearnData.inputs.Height, (uint)layerLearnData.inputs.Width }); //Used to be height, width

            inputsValues.SetData(layerLearnData.inputs.Values); // Used to be transpose

            nodeValuesValues.SetData(layerLearnData.nodeValues.Values);

            weightsValues.SetData(new double[costGradientWeightLength]);

            //Set Buffers
            computeShader.SetBuffer(0, "dimensions", dimensions);

            computeShader.SetBuffer(0, "inputs", inputsValues);
            computeShader.SetBuffer(0, "nodeValues", nodeValuesValues);
            computeShader.SetBuffer(0, "costGradient", weightsValues);

            //Calculate
            computeShader.Dispatch(0, _costGradientWeight.Width, _costGradientWeight.Height, 1);

            DNAMatrix costGradientWeight = new DNAMatrix(_costGradientWeight.Height, _costGradientWeight.Width);

            //Receive Result
            weightsValues.GetData(costGradientWeight.Values);

            //Clear Memory
            inputsValues.Release();
            nodeValuesValues.Release();
            weightsValues.Release();

            dimensions.Release();

            return costGradientWeight;
        }

        public void ParallelHiddenLayerNodeCalc(DNALayerLearnData[] layerLearnData, DNALayer oldLayer, DNAMatrix[] oldNodeValues)
        {
            int layerLearnDataLength = layerLearnData.Length * layerLearnData[0].weightedInputs.Length;
            int oldNodeValuesLength = oldNodeValues.Length * oldNodeValues[0].Length;
            int oldLayerWeightsLength = oldLayer.weights.Length;
            int nodeValuesLength = layerLearnData.Length * layerLearnData[0].nodeValues.Length;

            ComputeShader computeShader = ParallelHiddenLayerNode;

            ComputeBuffer dimensions = new ComputeBuffer(4, sizeof(int) * 2);
            ComputeBuffer oldLayerWeights = new ComputeBuffer(oldLayerWeightsLength, sizeof(double));
            ComputeBuffer oldNodeVals = new ComputeBuffer(oldNodeValuesLength, sizeof(double));
            ComputeBuffer weightedInputs = new ComputeBuffer(layerLearnDataLength, sizeof(double));
            ComputeBuffer nodeValues = new ComputeBuffer(nodeValuesLength, sizeof(double));
            ComputeBuffer activationDerivative = new ComputeBuffer(1, sizeof(int));

            //Set Data 
            dimensions.SetData(new int[] { oldLayer.weights.Height, oldLayer.weights.Width, oldNodeValues[0].Width, oldNodeValues[0].Height, layerLearnData[0].weightedInputs.Width, layerLearnData[0].weightedInputs.Height, layerLearnData[0].nodeValues.Width, layerLearnData[0].nodeValues.Height });
            oldLayerWeights.SetData(oldLayer.weights.Values); //Transpose
            oldNodeVals.SetData(OldNodeValArray(oldNodeValues));
            weightedInputs.SetData(GetWeightedInputs(layerLearnData));
            nodeValues.SetData(new double[nodeValuesLength]);
            activationDerivative.SetData(new int[] { this.activation.GetActivationFunctionIndex() });

            //Set Buffers
            computeShader.SetBuffer(0, "dimensions", dimensions);

            computeShader.SetBuffer(0, "weightedInputs", weightedInputs);
            computeShader.SetBuffer(0, "oldNodeValues", oldNodeVals);
            computeShader.SetBuffer(0, "oldLayerWeights", oldLayerWeights);
            computeShader.SetBuffer(0, "nodeValues", nodeValues);
            computeShader.SetBuffer(0, "activationDerivative", activationDerivative);

            //Calculate
            computeShader.Dispatch(0, layerLearnData[0].nodeValues.Width, layerLearnData[0].nodeValues.Height, layerLearnData.Length);

            double[] nodeVals = new double[nodeValuesLength];

            //Receive Result
            nodeValues.GetData(nodeVals);

            //Format Correctly
            SetHiddenNodeValues(nodeVals, layerLearnData);

            //Clear Memory
            weightedInputs.Release();
            activationDerivative.Release();
            oldLayerWeights.Release();
            oldNodeVals.Release();
            nodeValues.Release();
            dimensions.Release();
        }

        private DNAMatrix HiddenLayerNodeCalc(DNALayerLearnData layerLearnData, DNALayer oldLayer, DNAMatrix oldNodeValues)
        {
            int layerLearnDataLength = layerLearnData.weightedInputs.Length;
            int oldNodeValuesLength = oldNodeValues.Length * oldNodeValues.Length;
            int oldLayerWeightsLength = oldLayer.weights.Length;
            int nodeValuesLength = layerLearnData.nodeValues.Length;

            ComputeShader computeShader = HiddenLayerNode;

            //Oldlayer Weights, OldNodeVals, weightedInputs, derivative type, dimensions

            ComputeBuffer dimensions = new ComputeBuffer(4, sizeof(int) * 2);
            ComputeBuffer oldLayerWeights = new ComputeBuffer(oldLayerWeightsLength, sizeof(double));
            ComputeBuffer oldNodeVals = new ComputeBuffer(oldNodeValuesLength, sizeof(double));
            ComputeBuffer weightedInputs = new ComputeBuffer(layerLearnDataLength, sizeof(double));
            ComputeBuffer nodeValues = new ComputeBuffer(nodeValuesLength, sizeof(double));
            ComputeBuffer activationDerivative = new ComputeBuffer(1, sizeof(int));

            //Set Data                                         old Layer is switched because it is transposed
            dimensions.SetData(new int[] { oldLayer.weights.Height, oldLayer.weights.Width, oldNodeValues.Width, oldNodeValues.Height, layerLearnData.weightedInputs.Width, layerLearnData.weightedInputs.Height, layerLearnData.nodeValues.Width, layerLearnData.nodeValues.Height });
            oldLayerWeights.SetData(oldLayer.weights.Values);
            oldNodeVals.SetData(oldNodeValues.Values);
            weightedInputs.SetData(layerLearnData.weightedInputs.Values);
            nodeValues.SetData(new double[nodeValuesLength]);
            activationDerivative.SetData(new int[] { this.activation.GetActivationFunctionIndex() });

            //Set Buffers
            computeShader.SetBuffer(0, "dimensions", dimensions);
            computeShader.SetBuffer(0, "weightedInputs", weightedInputs);
            computeShader.SetBuffer(0, "oldNodeValues", oldNodeVals);
            computeShader.SetBuffer(0, "oldLayerWeights", oldLayerWeights);
            computeShader.SetBuffer(0, "nodeValues", nodeValues);
            computeShader.SetBuffer(0, "activationDerivative", activationDerivative);

            //Calculate
            computeShader.Dispatch(0, layerLearnData.nodeValues.Width, layerLearnData.nodeValues.Height, 1);

            //Receive Result
            double[] nodeVals = new double[nodeValuesLength];

            nodeValues.GetData(nodeVals);

            DNAMatrix matrix = new DNAMatrix(layerLearnData.nodeValues.Height, layerLearnData.nodeValues.Width);

            matrix.Values = nodeVals;

            //Clear Memory
            weightedInputs.Release();
            activationDerivative.Release();
            oldLayerWeights.Release();
            oldNodeVals.Release();
            nodeValues.Release();
            dimensions.Release();

            return matrix;
        }

        public double[] OldNodeValArray(DNAMatrix[] oldNodeVals)
        {
            double[] nodeVals = new double[oldNodeVals.Length * oldNodeVals[0].Length];

            int count = 0;
            foreach (DNAMatrix matrix in oldNodeVals)
            {
                Array.Copy(matrix.Values, 0, nodeVals, count, matrix.Length);
                count += matrix.Length;
            }

            return nodeVals;
        }

        private double[] WeightedInputs(DNALayerLearnData[] layerLearnData)
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

        private void SetHiddenNodeValues(double[] nodeValues, DNALayerLearnData[] layerLearnData)
        {
            for (int i = 0; i < layerLearnData.Length; i++)
            {
                Array.Copy(nodeValues, i * layerLearnData[i].nodeValues.Length, layerLearnData[i].nodeValues.Values, 0, layerLearnData[i].nodeValues.Length);
            }
        }
    }
}