using Google.Protobuf;
using MathNet.Numerics.LinearAlgebra;

class Program
{
    // static void Main()
    // {
    //     ScheduleCreation scheduleCreator = new ScheduleCreation(days: 1);
    //     int trainingNumber = 400;
    //     int testingNumber = 10; 
    //     Matrix<double> xTrainMat = scheduleCreator.generateSchedule(trainingNumber);
    //     Matrix<double> xTestMat = scheduleCreator.generateSchedule(testingNumber);

    //     List<NeuralNetwork> population = new List<NeuralNetwork>();

    //     for (int index = 0; index < 300; index++) {
    //         NeuralNetwork network = new NeuralNetwork(
    //             new List<IBaseLayer>{
    //                 new DenseLayer(scheduleCreator.segments, 124),
    //                 new ActivationTanh(),
    //                 new DenseLayer(124, scheduleCreator.segments),
    //                 new ActivationSigmoid(),
    //                 new DenseLayer(scheduleCreator.segments, 124),
    //                 new ActivationTanh(),
    //                 new DenseLayer(124, scheduleCreator.segments),
    //                 new ActivationSigmoid()
    //             }
    //         );
    //         population.Add(network);
    //     }

    //     GeneticAlgorithmCore geneticAlgorithmCore = new GeneticAlgorithmCore();
    //     population = geneticAlgorithmCore.trainGenetically(population, xTrainMat, 2500);

    //     foreach (NeuralNetwork network in population) {
    //         Console.WriteLine(network.fitnessScore);
    //         for (int i = 0; i < xTestMat.RowCount; i++){
    //             // Extract one sample as [1, 784]
    //             Matrix<double> inputSample = xTestMat.SubMatrix(i, 1, 0, xTestMat.ColumnCount);

    //             Matrix<double> output = network.predictOutcome(inputSample);
    //             Console.WriteLine($"Input: {inputSample}");
    //             Console.WriteLine($"Output: {output}");
    //             network.fitnessScore = GeneticAlgorithmCore.CalendarBasedFitness(inputSample.SubMatrix(i % inputSample.RowCount, 1, 0, inputSample.ColumnCount), output);
    //             Console.WriteLine($"Network Fitness Score: {network.fitnessScore}. Max Score: {scheduleCreator.segments}");
    //             Console.WriteLine(new string('-', 20));
    //         }
    //         Console.WriteLine("Done with network");
    //         break; // just break after the first network for demonstration
    //     }
    // }

    static void Main() {
        Random random = new Random();
        int mutationChance = 60; //% chance out of 100
        int scheduleSize = 96*14; //96 segments in a day. 1344 in 2 weeks
        int dataSetSize = 1;
        int populationSize = 5000;
        int generationLimit = 300;
        int immigrantCountPercent = 2;

        if (populationSize * immigrantCountPercent / 100 % 1 != 0) {
            throw new Exception("Population size must be divisible by immigrant count percent");
        }

        for (int i = 0; i < dataSetSize; i++) {
            // int taskSize = random.Next(1,13);
            Console.WriteLine("Enter task size: ");
            int taskSize = int.Parse(Console.ReadLine());
            GeneticAlgorithmGenerate geneticAlgorithm = new GeneticAlgorithmGenerate(
                scheduleSize, taskSize, populationSize, mutationChance, generationLimit, immigrantCountPercent/100 * populationSize);
            ScheduleBitMap topPerformer = geneticAlgorithm.TrainGenetically();
            printSchedule(topPerformer);
            Console.WriteLine($"Top performer fitness: {topPerformer.fitness} \nTask size: {taskSize} \nDeviation: {topPerformer.scheduleDeviation} \nTime elapsed: {geneticAlgorithm.geneticStopWatch.Elapsed}");
        }
    }

    public static void printSchedule(ScheduleBitMap schedule) {
        for (int i = 0; i < schedule.scheduleSize; i++) {
            if ((i) % 96 == 0) {
                Console.WriteLine($"Day: {(int)(i/96)}");
            }
            if (schedule.getBitValue(i)) {
                if (schedule.taskIndexes.Contains(i)) {
                    Console.Write("2");
                }
                else {
                    Console.Write("1");
                }
            }
            else {
                Console.Write("0");
            }
            if ((i+1) % 24 == 0) {
                Console.WriteLine();
            }
        }
    }
}
