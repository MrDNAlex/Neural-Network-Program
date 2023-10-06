using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using UnityEditor;


namespace DNAMath
{
    /// <summary>
    /// Custom Matrix Class developped for working on the GPU and with DNANeuralNetworks
    /// </summary>
    public class DNAMatrix
    {
        // 0--------> Width
        // |
        // |
        // |
        // Height


        public static ComputeShader shader;

        /// <summary>
        /// Describes the number of rows the matrix has
        /// </summary>
        private int _height;

        public int Height
        {
            get
            {
                return _height;
            }
        }

        /// <summary>
        /// Describes the number of columns the matrix has
        /// </summary>
        private int _width;

        public int Width
        {
            get
            {
                return _width;
            }
        }

        /// <summary>
        /// A list of all values contained in the matrix
        /// </summary>
        private double[] _values;

        public double[] Values
        {
            get
            {
                return _values;
            }
            set
            {
                _values = value;
            }
        }

        public DNAMatrix (int height, int width)
        {
            this._width = width;
            this._height = height;

            _values = new double[width * height];
        }

        /// <summary>
        /// Indexer allowing us to get access and sey  to a value using array notation
        /// </summary>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <returns></returns>
        public double this[int height, int width]
        {
            get
            {
                return _values[GetFlatIndex(height, width)];
            }
            set
            {
                _values[GetFlatIndex(height, width)] = value;
            }
        }

        public double this[int index]
        {
            get
            {
                
                return _values[index];
            }
        }


        /// <summary>
        /// Initializes a matrix that counts from 1 - the number of values based on the dimensions
        /// </summary>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <returns></returns>
        public static DNAMatrix Increment(int height, int width)
        {
            DNAMatrix matrix = new DNAMatrix(height, width);

            for (int i = 0; i < width * height; i ++)
            {
                matrix._values[i] = i;
            }
          
            return matrix;
        }

        /// <summary>
        /// Sets the value at the given height and width index
        /// </summary>
        /// <param name="heightIndex"></param>
        /// <param name="widthIndex"></param>
        /// <param name="val"></param>
        public void SetValue(int heightIndex, int widthIndex, double val)
        {
            this._values[GetFlatIndex(heightIndex, widthIndex)] = val;
        }

        /// <summary>
        /// Add to the values at the given height and width index
        /// </summary>
        /// <param name="heightIndex"></param>
        /// <param name="widthIndex"></param>
        /// <param name="val"></param>
        public void AddValue(int heightIndex, int widthIndex, double val)
        {
            this._values[GetFlatIndex(heightIndex, widthIndex)] += val;
        }

        /// <summary>
        /// Gets the value at the given height and width index
        /// </summary>
        /// <param name="heightIndex"></param>
        /// <param name="widthIndex"></param>
        /// <returns></returns>
        public double GetValue(int heightIndex, int widthIndex)
        {
            return _values[GetFlatIndex(heightIndex, widthIndex)];
        }

        /// <summary>
        /// Returns the flat index of a value 
        /// </summary>
        /// <param name="heightIndex"></param>
        /// <param name="widthIndex"></param>
        /// <returns></returns>
        public int GetFlatIndex(int heightIndex, int widthIndex)
        {
            return heightIndex * Width + widthIndex;
        }

        public static DNAMatrix operator +(DNAMatrix mat1, DNAMatrix mat2)
        {
            DNAMatrix newMat = new DNAMatrix(new Vector2Int(0, 0));

            if (mat1.matrixDimensions == mat2.matrixDimensions)
            {
                newMat = new DNAMatrix(mat1.matrixDimensions);

                for (int height = 0; height < newMat.matrixDimensions.x; height++)
                {
                    for (int width = 0; width < newMat.matrixDimensions.y; width++)
                    {
                        newMat.setValue(height, width, mat1.getValue(height, width) + mat2.getValue(height, width));
                    }
                }
            }
            else
            {
                Debug.Log("Error, Dimensions don't match");
            }

            return newMat;
        }

       

        /*
        struct GPUMatrix
        {
            public Vector2Int matrixDimensions;
            public float[] values;
        }

        public Vector2Int matrixDimensions;
        public double[] values;
        public static ComputeShader shader;

        public DNAMatrix(Vector2Int size)
        {
            //size.x = height
            //size.y = width
            this.matrixDimensions = size;
            this.values = new double[size.x * size.y];
        }

        public DNAMatrix(Vector2Int size, double[] values)
        {
            //size.x = height
            //size.y = width
            this.matrixDimensions = size;
            this.values = values;
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterAssembliesLoaded)]
        public static void loadAssets ()
        {
            shader = Resources.Load<ComputeShader>("Shade");
        }


        public static DNAMatrix Increment(Vector2Int size)
        {
            DNAMatrix matrix = new DNAMatrix(size);

            for (int matHeight = 0; matHeight < matrix.matrixDimensions.x; matHeight++)
            {
                for (int matWidth = 0; matWidth < matrix.matrixDimensions.y; matWidth++)
                {
                    matrix.setValue(matHeight, matWidth, matHeight * matrix.matrixDimensions.y + matWidth + 1);
                }
            }

            //Debug.Log(displayMat(matrix));
            return matrix;

        }

        static double dotProduct(DNAMatrix mat1, DNAMatrix mat2, int hIndex, int wIndex)
        {
            double result = 0;

            for (int i = 0; i < mat1.matrixDimensions.y; i++)
            {
                //Debug.Log("i: " + i);
                // Debug.Log("w: " + wIndex);
                // Debug.Log("h: " + hIndex);
                result += mat1.getValue(hIndex, i) * mat2.getValue(i, wIndex);
            }

            return result;
        }

       public static DNAMatrix operator +(DNAMatrix mat1, DNAMatrix mat2)
        {
            DNAMatrix newMat = new DNAMatrix(new Vector2Int(0, 0));

            if (mat1.matrixDimensions == mat2.matrixDimensions)
            {
                newMat = new DNAMatrix(mat1.matrixDimensions);

                for (int height = 0; height < newMat.matrixDimensions.x; height++)
                {
                    for (int width = 0; width < newMat.matrixDimensions.y; width++)
                    {
                        newMat.setValue(height, width, mat1.getValue(height, width) + mat2.getValue(height, width));
                    }
                }
            } else
            {
                Debug.Log("Error, Dimensions don't match");
            }

            return newMat;
        }

        public static DNAMatrix operator -(DNAMatrix mat1, DNAMatrix mat2)
        {
            DNAMatrix newMat = new DNAMatrix(new Vector2Int(0, 0));

            if (mat1.matrixDimensions == mat2.matrixDimensions)
            {
                newMat = new DNAMatrix(mat1.matrixDimensions);

                for (int height = 0; height < newMat.matrixDimensions.x; height++)
                {
                    for (int width = 0; width < newMat.matrixDimensions.y; width++)
                    {
                        newMat.setValue(height, width, mat1.getValue(height, width) - mat2.getValue(height, width));
                    }
                }
            }
            else
            {
                Debug.Log("Error, Dimensions don't match");
            }

            return newMat;
        }

        public static DNAMatrix operator *(DNAMatrix mat1, DNAMatrix mat2)
        {
            DNAMatrix newMat = new DNAMatrix(new Vector2Int(0, 0));

            //Check if mat1.y == mat2.x //aka width == height
            if (mat1.matrixDimensions.y == mat2.matrixDimensions.x)
            {
                Vector2Int dim;

                dim = new Vector2Int(mat1.matrixDimensions.x, mat2.matrixDimensions.y);

                newMat = new DNAMatrix(dim);

               // Debug.Log(displayMat(newMat));

                //Set values
                if (dim.x >= dim.y)
                {
                    //x = height   y = width
                    //If height is bigger than width 

                    //loop over height so that more GPU cores are used

                    for (int height = 0; height < dim.x; height++)
                    {
                        for (int width = 0; width < dim.y; width++)
                        {
                            newMat.setValue(height, width, dotProduct(mat1, mat2, height, width));
                        }
                    }
                }
                else
                {
                    //x = height   y = width
                    //If height is bigger than width 

                    //loop over height so that more GPU cores are used

                    for (int width = 0; width < dim.y; width++)
                    {
                        for (int height = 0; height < dim.x; height++)
                        {
                            newMat.setValue(height, width, dotProduct(mat1, mat2, height, width));
                        }
                    }
                }

                //Debug.Log(displayMat(newMat));

                
            }
            else
            {
                Debug.Log("Error, Dimensions don't match");
            }

            return newMat;
        }

        public void setValue(int hIndex, int wIndex, double val)
        {
            this.values[GetFlatIndex(hIndex, wIndex)] = val;
        }

        public void addValue (int hIndex, int wIndex, double val)
        {
            this.values[GetFlatIndex(hIndex, wIndex)] += val;
        }

        public double getValue(int hIndex, int wIndex)
        {
            return values[GetFlatIndex(hIndex, wIndex)];
        }

        public int GetFlatIndex(int hIndex, int wIndex)
        {
            return hIndex * matrixDimensions.y + wIndex;
        }

        public static string displayMat(DNAMatrix mat)
        {
            //Display the matrix
            string line = "\n";
            for (int height = 0; height < mat.matrixDimensions.x; height++)
            {
                for (int width = 0; width < mat.matrixDimensions.y; width++)
                {

                    //Debug.Log("Width: " + width + " Height: " + height + " = " + newMat[getIndex(width, height, dim.x)]);

                    line += mat.getValue(height, width) + "    ";
                }
                line += "\n";

            }
            return line;
        }

        public DNAMatrix multMatrixGPU( DNAMatrix mat1, DNAMatrix mat2)
        {
            DNAMatrix newMatrix = new DNAMatrix(new Vector2Int(0, 0));
            if (mat1.matrixDimensions.y == mat2.matrixDimensions.x)
            {
                Vector2Int dim;

                dim = new Vector2Int(mat1.matrixDimensions.x, mat2.matrixDimensions.y);

                newMatrix = new DNAMatrix(dim);

                ComputeShader compShader = shader;

                ComputeBuffer mat1Vals = new ComputeBuffer(mat1.values.Length, sizeof(double));
                mat1Vals.SetData(mat1.values);

                ComputeBuffer mat2Vals = new ComputeBuffer(mat2.values.Length, sizeof(double));
                mat2Vals.SetData(mat2.values);

                ComputeBuffer newMatVals = new ComputeBuffer(newMatrix.values.Length, sizeof(double));
                newMatVals.SetData(newMatrix.values);

                //Convert dimensions to array of int
                int[] dim1 = new int[2];
                dim1[0] = mat1.matrixDimensions.x;
                dim1[1] = mat1.matrixDimensions.y;
                ComputeBuffer mat1Dim = new ComputeBuffer(2, sizeof(int));
                mat1Dim.SetData(dim1);

                int[] dim2 = new int[2];
                dim2[0] = mat2.matrixDimensions.x;
                dim2[1] = mat2.matrixDimensions.y;
                ComputeBuffer mat2Dim = new ComputeBuffer(2, sizeof(int));
                mat2Dim.SetData(dim2);

                int[] newDim = new int[2];
                newDim[0] = newMatrix.matrixDimensions.x;
                newDim[1] = newMatrix.matrixDimensions.y;
                ComputeBuffer newMatDim = new ComputeBuffer(2, sizeof(int));
                newMatDim.SetData(newDim);

                //Set the buffer info
                compShader.SetBuffer(0, "mat1Vals", mat1Vals);
                compShader.SetBuffer(0, "mat2Vals", mat2Vals);
                compShader.SetBuffer(0, "newMatVals", newMatVals);
                compShader.SetBuffer(0, "mat1Dim", mat1Dim);
                compShader.SetBuffer(0, "mat2Dim", mat2Dim);
                compShader.SetBuffer(0, "newMatDim", newMatDim);

                System.DateTime startTime;
                System.DateTime endTime;

                startTime = System.DateTime.UtcNow;

                int groupCount = 0;
                if ((newMatrix.values.Length / 1024) >= 1)
                {
                    groupCount = (newMatrix.values.Length / 1024);
                }
                else
                {
                    groupCount = 1;
                }

                compShader.Dispatch(0, groupCount, 1, 1);

               

                //Calculate
               

                newMatVals.GetData(newMatrix.values);

                //Debug.Log("New Mat:" + displayMat(newMatrix));

                mat1Vals.Dispose();
                mat2Vals.Dispose();
                newMatVals.Dispose();

                mat1Dim.Dispose();
                mat2Dim.Dispose();
                newMatDim.Dispose();

                endTime = System.DateTime.UtcNow;

                Debug.Log("Total Time elapsed GPU (Matrix Multiplication) (" + mat1.matrixDimensions.x + "x" + mat1.matrixDimensions.y + "*" + mat2.matrixDimensions.x + "x" + mat2.matrixDimensions.y + "): " + (endTime - startTime).TotalMilliseconds + "(MilliSeconds)");
            }

            return newMatrix;
        }
        */
    }
}

