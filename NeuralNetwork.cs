using MathNet.Numerics.LinearAlgebra;

public class NeuralNetwork {
    public List<IBaseLayer> layers {get; set;}
    public double fitnessScore {get; set;} = 0;
    public NeuralNetwork(List<IBaseLayer> layers) {
        this.layers = layers;
    }

    public Matrix<double> predictOutcome(Matrix<double> input) {
        Matrix<double> output = input;
        foreach (IBaseLayer layer in layers) {
            output = layer.ForwardProp(output);
        }
        return output;
    }
}