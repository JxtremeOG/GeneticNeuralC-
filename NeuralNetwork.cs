using MathNet.Numerics.LinearAlgebra;

public class NeuralNetwork {
    public List<IBaseLayer> layers {get; set;}
    public double fitnessScore {get; set;} = 0;
    public NeuralNetwork(List<IBaseLayer> layers) {
        this.layers = layers;
    }

    public double MeanSquaredError(Matrix<double> yTrue, Matrix<double> yPred)
    {
        Matrix<double> diff = yTrue - yPred;  // element-wise difference
        Matrix<double> squared = diff.PointwisePower(2.0);
        return squared.Enumerate().Average(); // or squared.RowSums().Sum() / squared.Count
    }

    public Matrix<double> MeanSquaredErrorPrime(Matrix<double> yTrue, Matrix<double> yPred)
    {
        // derivative = 2 * (yPred - yTrue) / numElements
        Matrix<double> diff = (yPred - yTrue) * 2.0;
        double numElements = diff.RowCount * diff.ColumnCount;
        return diff / numElements;
    }

    public Matrix<double> predictOutcome(Matrix<double> input) {
        Matrix<double> output = input;
        foreach (IBaseLayer layer in layers) {
            output = layer.ForwardProp(output);
        }
        return output;
    }
    
    public void Train(
        Func<Matrix<double>, Matrix<double>, double> loss,
        Func<Matrix<double>, Matrix<double>, Matrix<double>> lossPrime,
        List<Matrix<double>> xTrain,
        List<Matrix<double>> yTrain,
        int epochs = 1000,
        double learningRate = 0.01,
        bool verbose = true)
    {
        for (int e = 0; e < epochs; e++)
        {
            double error = 0.0;

            // Assume xTrain.Count == yTrain.Count
            for (int i = 0; i < xTrain.Count; i++)
            {
                var x = xTrain[i];
                var y = yTrain[i];

                // forward
                var output = predictOutcome(x);

                // accumulate error
                error += loss(y, output);

                // backward
                var grad = lossPrime(y, output);
                // traverse network in reverse
                for (int layerIndex = layers.Count - 1; layerIndex >= 0; layerIndex--)
                {
                    grad = layers[layerIndex].BackwardProp(grad, learningRate);
                }
            }

            // average error over the dataset
            error /= xTrain.Count;

            if (verbose)
            {
                Console.WriteLine($"{e + 1}/{epochs}, error={error}");
            }
        }
    }

}