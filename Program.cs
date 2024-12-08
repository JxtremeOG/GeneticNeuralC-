using MathNet.Numerics.LinearAlgebra;

class Program
{
    static void Main()
    {
        // Paths to MNIST data files
        string dataDir = @"C:\Users\zoldw\Projects\Coding\Learning Projects\GeneticNeuralC#\MNISTData";
        string trainImagesPath = Path.Combine(dataDir, "train-images-idx3-ubyte");
        string trainLabelsPath = Path.Combine(dataDir, "train-labels-idx1-ubyte");
        string testImagesPath = Path.Combine(dataDir, "t10k-images-idx3-ubyte");
        string testLabelsPath = Path.Combine(dataDir, "t10k-labels-idx1-ubyte");

        // Just load as byte arrays
        var (xTrainByte, yTrainByte) = LoadMNIST(trainImagesPath, trainLabelsPath);
        var (xTestByte, yTestByte) = LoadMNIST(testImagesPath, testLabelsPath);

        // Preprocess immediately - this will give shapes similar to Python:
        // xTrain: [1000, 784, 1], yTrain: [1000, 10, 1]
        var (xTrain, yTrain) = PreprocessData(xTrainByte, yTrainByte, 1000);
        var (xTest, yTest) = PreprocessData(xTestByte, yTestByte, 20);

        Console.WriteLine($"xTrain shape: {xTrain.GetLength(0)}, {xTrain.GetLength(1)}, {xTrain.GetLength(2)}");
        Console.WriteLine($"yTrain shape: {yTrain.GetLength(0)}, {yTrain.GetLength(1)}, {yTrain.GetLength(2)}");
        Console.WriteLine($"xTest shape: {xTest.GetLength(0)}, {xTest.GetLength(1)}, {xTest.GetLength(2)}");
        Console.WriteLine($"yTest shape: {yTest.GetLength(0)}, {yTest.GetLength(1)}, {yTest.GetLength(2)}");

        // Convert 3D arrays ([samples, width, 1]) to 2D matrices ([samples, width])
        // This matches the Python final format where samples are rows
        Matrix<double> xTrainMat = ConvertToMatrix(xTrain);        // [1000, 784]
        Matrix<double> yTrainMat = ConvertToMatrix(yTrain, 10);    // [1000, 10]
        Matrix<double> xTestMat = ConvertToMatrix(xTest);          // [20, 784]
        Matrix<double> yTestMat = ConvertToMatrix(yTest, 10);      // [20, 10]

        List<NeuralNetwork> population = new List<NeuralNetwork>();

        for (int index = 0; index < 400; index++) {
            NeuralNetwork network = new NeuralNetwork(
                new List<IBaseLayer>{
                    new DenseLayer(28 * 28, 40),
                    new ActivationTanh(),
                    new DenseLayer(40, 10),
                    new ActivationTanh()
                }
            );
            population.Add(network);
        }

        GeneticAlgorithmCore geneticAlgorithmCore = new GeneticAlgorithmCore();
        population = geneticAlgorithmCore.trainGenetically(population, xTrainMat, yTrainMat, 2000);

        foreach (NeuralNetwork network in population) {
            Console.WriteLine(network.fitnessScore);
            for (int i = 0; i < xTestMat.RowCount; i++){
                // Extract one sample as [1, 784]
                Matrix<double> inputSample = xTestMat.SubMatrix(i, 1, 0, xTestMat.ColumnCount);
                Matrix<double> trueLabel = yTestMat.SubMatrix(i, 1, 0, yTestMat.ColumnCount);

                Matrix<double> output = network.predictOutcome(inputSample);
                Console.WriteLine($"Prediction: {getMax(output, false):0}, True: {getMax(trueLabel, false)}");
                double error = geneticAlgorithmCore.MeanSquaredError(trueLabel, output);
                Console.WriteLine($"Error: {Math.Round(error, 3) * 100:0.0000}%");
                Console.WriteLine(new string('-', 20));
            }
            Console.WriteLine("Done with network");
            break; // just break after the first network for demonstration
        }
    }

    static double getMax(Matrix<double> matrix, bool value = true) {
        int maxIndex = 0;
        double maxValue = matrix[0, 0];

        for (int j = 0; j < matrix.ColumnCount; j++)
        {
            if (matrix[0, j] > maxValue)
            {
                maxValue = matrix[0, j];
                maxIndex = j;
            }
        }
        if (value)
        {
            return maxValue;
        }
        return maxIndex;
    }

    static (float[,,] x, float[,,] y) PreprocessData(byte[,,] x, byte[,] y, int limit)
    {
        int samples = x.GetLength(0);
        int height = x.GetLength(1);
        int width = x.GetLength(2);

        // reshape and normalize input data
        int newDim = width * height;
        float[,,] xFloat = new float[samples, newDim, 1];
        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < height; j++)
            {
                for (int k = 0; k < width; k++)
                {
                    xFloat[i, j * width + k, 0] = x[i, j, k] / 255f;
                }
            }
        }

        // one-hot encode output
        float[,,] yOneHot = ToCategorical(y);

        limit = Math.Min(limit, samples);

        float[,,] xLimited = new float[limit, newDim, 1];
        float[,,] yLimited = new float[limit, 10, 1];

        for (int i = 0; i < limit; i++)
        {
            for (int j = 0; j < newDim; j++)
            {
                xLimited[i, j, 0] = xFloat[i, j, 0];
            }
            for (int c = 0; c < 10; c++)
            {
                yLimited[i, c, 0] = yOneHot[i, c, 0];
            }
        }

        return (xLimited, yLimited);
    }

    static float[,,] ToCategorical(byte[,] labels)
    {
        int samples = labels.GetLength(0);
        float[,,] oneHot = new float[samples, 10, 1];
        for (int i = 0; i < samples; i++)
        {
            byte label = labels[i, 0];
            oneHot[i, label, 0] = 1f;
        }
        return oneHot;
    }

    static (byte[,,] x, byte[,] y) LoadMNIST(string imagePath, string labelPath)
    {
        byte[] labels = ReadLabels(labelPath, out int numLabels);
        byte[][] images = ReadImages(imagePath, numLabels, out int rows, out int cols);

        byte[,,] xData = new byte[numLabels, rows, cols];
        byte[,] yData = new byte[numLabels, 1];

        for (int i = 0; i < numLabels; i++)
        {
            yData[i, 0] = labels[i];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    xData[i, r, c] = images[i][r * cols + c];
                }
            }
        }

        return (xData, yData);
    }

    static byte[] ReadLabels(string labelPath, out int numLabels)
    {
        using (var fs = new FileStream(labelPath, FileMode.Open, FileAccess.Read))
        using (var br = new BinaryReader(fs))
        {
            int magic = ReadBigEndianInt32(br);
            numLabels = ReadBigEndianInt32(br);
            return br.ReadBytes(numLabels);
        }
    }

    static byte[][] ReadImages(string imagePath, int expectedCount, out int rows, out int cols)
    {
        using (var fs = new FileStream(imagePath, FileMode.Open, FileAccess.Read))
        using (var br = new BinaryReader(fs))
        {
            int magic = ReadBigEndianInt32(br);
            int numImages = ReadBigEndianInt32(br);
            rows = ReadBigEndianInt32(br);
            cols = ReadBigEndianInt32(br);

            if (numImages != expectedCount)
            {
                throw new Exception("Mismatch between label and image count.");
            }

            byte[][] images = new byte[numImages][];
            for (int i = 0; i < numImages; i++)
            {
                images[i] = br.ReadBytes(rows * cols);
            }
            return images;
        }
    }

    static int ReadBigEndianInt32(BinaryReader br)
    {
        byte[] bytes = br.ReadBytes(4);
        if (BitConverter.IsLittleEndian)
            Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }

    static Matrix<double> ConvertToMatrix(double[,,] array3D)
    {
        int samples = array3D.GetLength(0);
        int width = array3D.GetLength(1);
        var mat = Matrix<double>.Build.Dense(samples, width);
        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < width; j++)
            {
                mat[i, j] = array3D[i, j, 0];
            }
        }
        return mat;
    }

    static Matrix<double> ConvertToMatrix(float[,,] array3D)
    {
        int samples = array3D.GetLength(0);
        int width = array3D.GetLength(1);
        var mat = Matrix<double>.Build.Dense(samples, width);
        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < width; j++)
            {
                mat[i, j] = (double)array3D[i, j, 0];
            }
        }
        return mat;
    }

    static Matrix<double> ConvertToMatrix(double[,,] array3D, int classes)
    {
        int samples = array3D.GetLength(0);
        var mat = Matrix<double>.Build.Dense(samples, classes);
        for (int i = 0; i < samples; i++)
        {
            for (int c = 0; c < classes; c++)
            {
                mat[i, c] = array3D[i, c, 0];
            }
        }
        return mat;
    }

    static Matrix<double> ConvertToMatrix(float[,,] array3D, int classes)
    {
        int samples = array3D.GetLength(0);
        var mat = Matrix<double>.Build.Dense(samples, classes);
        for (int i = 0; i < samples; i++)
        {
            for (int c = 0; c < classes; c++)
            {
                mat[i, c] = (double)array3D[i, c, 0];
            }
        }
        return mat;
    }
}
