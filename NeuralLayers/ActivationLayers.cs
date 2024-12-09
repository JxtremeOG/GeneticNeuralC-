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
    public IBaseLayer cloneLayer()
    {
        return new ActivationSigmoid();
    }
}