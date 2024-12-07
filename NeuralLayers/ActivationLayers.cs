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
}