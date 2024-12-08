using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

public class DenseLayer : IBaseLayer {
    public Matrix<double> weights { get; set; }
    public Matrix<double> bias { get; set; }
    public Matrix<double>? Input { get; set; }
    public Matrix<double>? Output { get; set; }

    public static Random denseRandom = new Random();
    private static readonly ThreadLocal<Random> ThreadRandom = new(() => new Random(Guid.NewGuid().GetHashCode()));

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

    public void mutateWeights(double mutationRate, List<double> mutationRange, double mutationMultiplier) {
        mutationRate = adjustMutationRate(mutationRate, mutationMultiplier);
        if (mutationRate < 0 || mutationRate > 1) {
            throw new ArgumentException("Invalid mutation rate");
        }

        double minRange = mutationRange.Min();
        double maxRange = mutationRange.Max();
        double rangeSpan = maxRange - minRange;

        int rowCount = weights.RowCount;
        int columnCount = weights.ColumnCount;
        
        // Parallelize the outer loop to utilize multiple cores
        Parallel.For(0, rowCount, i =>
        {
            var rand = ThreadRandom.Value;
            for (int j = 0; j < columnCount; j++)
            {
                if (rand!.NextDouble() < mutationRate)
                {
                    double rangeValue = minRange + (rand.NextDouble() * rangeSpan);
                    weights[i, j] += weights[i, j] * rangeValue;
                }
            }
        });
    }

    private double adjustMutationRate(double mutationRate, double mutationMultiplier) {
        mutationRate *= mutationMultiplier;
        mutationRate = Math.Abs(mutationRate);
        if (mutationRate > 1) {
            mutationRate = .9;
        }
        return mutationRate;
    }

    public void mutateBias(double mutationRate, List<double> mutationRange, double mutationMultiplier) {
        mutationRate = adjustMutationRate(mutationRate, mutationMultiplier);
        if (mutationRate < 0 || mutationRate > 1) {
            throw new ArgumentException("Invalid mutation rate");
        }

        double minRange = mutationRange.Min();
        double maxRange = mutationRange.Max();
        double rangeSpan = maxRange - minRange;

        int rowCount = bias.RowCount;
        int columnCount = bias.ColumnCount;

        Parallel.For(0, rowCount, i =>
        {
            var rand = ThreadRandom.Value;
            for (int j = 0; j < columnCount; j++)
            {
                if (rand!.NextDouble() < mutationRate)
                {
                    double rangeValue = minRange + (rand.NextDouble() * rangeSpan);
                    bias[i, j] += bias[i, j] * rangeValue;
                }
            }
        });
    }
}