using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

public class DenseLayer : IBaseLayer {
    public Matrix<double> weights { get; set; }
    public Matrix<double> bias { get; set; }
    public Matrix<double>? Input { get; set; }
    public Matrix<double>? Output { get; set; }

    public static Random denseRandom = new Random();

    public DenseLayer(int inputSize, int outputSize) {
        Normal normalDist = new Normal(0, 1);
        weights = Matrix<double>.Build.Dense(inputSize, outputSize, (r, c) => normalDist.Sample());
        bias = Matrix<double>.Build.Dense(1, outputSize, (r, c) => normalDist.Sample());
    }
    public Matrix<double> ForwardProp(Matrix<double> input)
    {
        this.Input = input;
        return input * weights + bias;
    }

    public (Matrix<double> LeftWeights, Matrix<double> RightWeights) splitWeights(int splitPoint)
    {
        // Validate splitPoint for rows
        if (splitPoint <= 0 || splitPoint >= weights.RowCount)
        {
            throw new ArgumentException($"Invalid split point: {splitPoint}. Must be between 1 and {weights.RowCount - 1}");
        }

        // Split the weights matrix by rows
        var leftWeights = weights.SubMatrix(0, splitPoint, 0, weights.ColumnCount);
        var rightWeights = weights.SubMatrix(splitPoint, weights.RowCount - splitPoint, 0, weights.ColumnCount);

        return (leftWeights, rightWeights);
    }


    public (Matrix<double> TopWeights, Matrix<double> LowerWeights) splitBias(int splitPoint)
    {
        // Validate splitPoint for rows
        if (!(0 < splitPoint && splitPoint < this.bias.RowCount))
        {
            throw new ArgumentException($"Invalid split point: {splitPoint}. Must be between 1 and {this.bias.RowCount - 1}");
        }

        // Split the bias matrix by rows
        Matrix<double> leftBias = this.bias.SubMatrix(0, splitPoint, 0, this.bias.ColumnCount);
        Matrix<double> rightBias = this.bias.SubMatrix(splitPoint, this.bias.RowCount - splitPoint, 0, this.bias.ColumnCount);

        return (leftBias, rightBias);
    }

    public void mutateWeights(double mutationRate, List<double> mutationRange) {
        if (mutationRate < 0 || mutationRate > 1) {
            throw new ArgumentException("Invalid mutation rate");
        }
        
        for (int i = 0; i < weights.RowCount; i++)
        {
            for (int j = 0; j < weights.ColumnCount; j++)
            {
                if (denseRandom.NextDouble() < mutationRate)
                {
                    double rangeValue = mutationRange.Min() + (denseRandom.NextDouble() * (mutationRange.Max() - mutationRange.Min()));
                    weights[i, j] += weights[i, j] * rangeValue;
                }
            }
        }
    }

    public void mutateBias(double mutationRate, List<double> mutationRange) {
        if (mutationRate < 0 || mutationRate > 1) {
            throw new ArgumentException("Invalid mutation rate");
        }
        
        for (int i = 0; i < bias.RowCount; i++)
        {
            for (int j = 0; j < bias.ColumnCount; j++)
            {
                if (denseRandom.NextDouble() < mutationRate)
                {
                    double rangeValue = mutationRange.Min() + (denseRandom.NextDouble() * (mutationRange.Max() - mutationRange.Min()));
                    bias[i, j] += bias[i, j] * rangeValue;
                }
            }
        }
    }
}