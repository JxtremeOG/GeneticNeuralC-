using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;

public class ActivationTanh : IBaseLayer
{
    public Matrix<double>? Input { get; set; }
    public Matrix<double>? Output { get; set; }

    public Matrix<double> ForwardProp(Matrix<double> input)
    {
        Input = input;
        Output = Input.Map(Math.Tanh);
        return Output;
    }
    public Matrix<double> BackwardProp(Matrix<double> outputGradient, double learningRate)
    {
        return outputGradient.PointwiseMultiply(1 - Output.PointwisePower(2));
    }
    public IBaseLayer cloneLayer()
    {
        return new ActivationTanh();
    }
}

public class ActivationSigmoid : IBaseLayer 
{
    public Matrix<double>? Input { get; set; }
    public Matrix<double>? Output { get; set; }
    public Matrix<double> ForwardProp(Matrix<double> input)
    {
        Input = input;
        Output = Input.Map(weightedInput => 1.0 / (1.0 + Math.Exp(-weightedInput)));
        return Output;
    }
    public Matrix<double> BackwardProp(Matrix<double> outputGradient, double learningRate)
    {
        // derivative of Sigmoid = Output .* (1 - Output)
        // chain rule => outputGradient .* derivative
        var derivative = Output.PointwiseMultiply(1 - Output);
        return outputGradient.PointwiseMultiply(derivative);
    }
    public IBaseLayer cloneLayer()
    {
        return new ActivationSigmoid();
    }
}