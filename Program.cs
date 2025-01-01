using System.Diagnostics;
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
        BinaryFileHandler fileHandler = new BinaryFileHandler();
        Console.WriteLine("1. Write Data or 2. Read Data");
        if (int.Parse(Console.ReadLine()) == 1) {
            Stopwatch primaryStopWatch = Stopwatch.StartNew();
            Console.WriteLine("Enter data set size: ");
            Random random = new Random();
            int mutationChance = 60; //% chance out of 100
            int scheduleSize = 96*14; //96 segments in a day. 1344 in 2 weeks
            int dataSetSize = int.Parse(Console.ReadLine());
            int populationSize = 5000;
            int generationLimit = 200; //Adaptive generation limit (10 * taskSize + generationLimit)
            int immigrantCountPercent = 2;
            int dynamicGenerationLimit;

            if (populationSize * immigrantCountPercent / 100 % 1 != 0) {
                throw new Exception("Population size must be divisible by immigrant count percent");
            }

            primaryStopWatch.Start();
            for (int i = 0; i < dataSetSize; i++) {
                Console.WriteLine($"Beginning training of data example {i}...");
                int taskSize = random.Next(1,13);
                // Console.WriteLine("Enter task size: ");
                // int taskSize = int.Parse(Console.ReadLine());
                double immigrantCount = populationSize * immigrantCountPercent / 100;
                dynamicGenerationLimit = generationLimit + taskSize * 10 > 300 ? 300 : generationLimit + taskSize * 10; //max out generation limit at 300
                GeneticAlgorithmGenerate geneticAlgorithm = new GeneticAlgorithmGenerate(
                    scheduleSize, taskSize, populationSize, mutationChance, dynamicGenerationLimit, (int)immigrantCount);
                ScheduleBitMap topPerformer = geneticAlgorithm.TrainGenetically().Result;
                // printSchedule(topPerformer);
                Console.WriteLine($"Top performer fitness: {topPerformer.fitness} \nTask size: {taskSize} \nGeneration Limit: {dynamicGenerationLimit} \nTime elapsed: {geneticAlgorithm.geneticStopWatch.Elapsed}");
                string binaryData = fileHandler.SaveDataInfo(topPerformer, geneticAlgorithm);
                byte[] binaryBytes = fileHandler.BinaryStringToByteArray(binaryData);
                fileHandler.SaveToBinaryFile(binaryBytes, "ScheduleData/calendarProbabilityTrainData.bin");
            }
            primaryStopWatch.Stop();
            Console.WriteLine($"Training for {dataSetSize} data examples elapsed over {primaryStopWatch.Elapsed}");
        }
        else {
            var data = fileHandler.FileToDataSet("ScheduleData/calendarProbabilityTrainData.bin");
            List<NeuralScheduleHandler> scheduleDataSet = data.Item1;
            List<Matrix<double>> inputDataSet = data.Item2;
            List<Matrix<double>> outputDataSet = data.Item3;
            Console.WriteLine($"{scheduleDataSet.Count} data examples read");
            Console.WriteLine($"Train Model? Y or N");
            if (Console.ReadLine() == "N") {
                return;
            }

            NeuralNetwork network = new NeuralNetwork(
                new List<IBaseLayer>{
                    new DenseLayer(1391, 1363),
                    new ActivationTanh(),
                    new DenseLayer(1363, 1356),
                    new ActivationSigmoid(),
                    new DenseLayer(1356, 1344),
                    new ActivationTanh(),
                    new DenseLayer(1344, 1344),
                    new ActivationSigmoid()
                }
            );

            Console.WriteLine($"Enter number of epochs: ");
            int epochs = int.Parse(Console.ReadLine());

            Console.WriteLine("Training model...");
            network.Train(network.MeanSquaredError, network.MeanSquaredErrorPrime, inputDataSet, outputDataSet, epochs, 0.1, true);

            var dataTest = fileHandler.FileToDataSet("ScheduleData/calendarProbabilityTestData.bin");
            List<NeuralScheduleHandler> scheduleTestDataSet = dataTest.Item1;
            List<Matrix<double>> inputTestDataSet = dataTest.Item2;
            List<Matrix<double>> outputTestDataSet = dataTest.Item3;

            for (int i = 0; i < inputTestDataSet.Count; i++) {
                Matrix<double> input = inputTestDataSet[i];
                Matrix<double> output = network.predictOutcome(input);
                var inputArray = input.ToArray();
                var outputArray = output.ToArray();

                Console.WriteLine($"Data Example {i} task size: {scheduleTestDataSet[i].taskSegments.Cast<bool>().Count(bit => bit)}");
                Console.WriteLine($"| exp  | out  | exp  | out  | exp  | out  | exp  | out  | exp  | out  | exp  | out  | exp  | out  | exp  | out  |");

                for (int j = 0; j < outputArray.Length; j+=8) {
                    for (int k = j; k < j+8; k++) {
                        Console.Write($"| {outputTestDataSet[i][0,k].ToString("0.00")} | {outputArray[0,k].ToString("0.00")} ");
                    }
                    Console.WriteLine("|");
                    if (j%96 == 0 && j != 0) {
                        Console.WriteLine(new string('-', 113));
                    }
                }
                Console.WriteLine();
                Console.ReadLine();
            }
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
