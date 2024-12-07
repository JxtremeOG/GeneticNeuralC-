using System;
using MathNet.Numerics.LinearAlgebra;

public interface IBaseLayer
{
    Matrix<double>? Input { get; set; }
    Matrix<double>? Output { get; set; }

    Matrix<double> ForwardProp(Matrix<double> input);
}