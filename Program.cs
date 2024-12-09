using MathNet.Numerics.LinearAlgebra;

class Program
{
    static void Main()
    {
        ScheduleCreation scheduleCreator = new ScheduleCreation(days: 1);
        int trainingNumber = 400;
        int testingNumber = 10; 
        Matrix<double> xTrainMat = scheduleCreator.generateSchedule(trainingNumber);
        Matrix<double> xTestMat = scheduleCreator.generateSchedule(testingNumber);

        List<NeuralNetwork> population = new List<NeuralNetwork>();

        for (int index = 0; index < 400; index++) {
            NeuralNetwork network = new NeuralNetwork(
                new List<IBaseLayer>{
                    new DenseLayer(scheduleCreator.segments, 124),
                    new ActivationTanh(),
                    new DenseLayer(124, scheduleCreator.segments),
                    new ActivationSigmoid(),
                    new DenseLayer(scheduleCreator.segments, 124),
                    new ActivationTanh(),
                    new DenseLayer(124, scheduleCreator.segments),
                    new ActivationSigmoid()
                }
            );
            population.Add(network);
        }

        GeneticAlgorithmCore geneticAlgorithmCore = new GeneticAlgorithmCore();
        population = geneticAlgorithmCore.trainGenetically(population, xTrainMat, 5000);

        foreach (NeuralNetwork network in population) {
            Console.WriteLine(network.fitnessScore);
            for (int i = 0; i < xTestMat.RowCount; i++){
                // Extract one sample as [1, 784]
                Matrix<double> inputSample = xTestMat.SubMatrix(i, 1, 0, xTestMat.ColumnCount);

                Matrix<double> output = network.predictOutcome(inputSample);
                Console.WriteLine($"Input: {inputSample}");
                Console.WriteLine($"Output: {output}");
                network.fitnessScore = GeneticAlgorithmCore.CalendarBasedFitness(inputSample.SubMatrix(i % inputSample.RowCount, 1, 0, inputSample.ColumnCount), output);
                Console.WriteLine($"Network Fitness Score: {network.fitnessScore}. Max Score: {scheduleCreator.segments}");
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
}
