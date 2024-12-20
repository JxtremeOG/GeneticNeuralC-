using System.Collections.Concurrent;
using System.Net.NetworkInformation;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

public class GeneticAlgorithmCore{

    private static readonly Random randomStatic = new Random();
    public double ErrorBasedFitness(Matrix<double> yExpect, Matrix<double> yOutput) {
        if (yExpect.RowCount != yOutput.RowCount || yExpect.ColumnCount != yOutput.ColumnCount){
            throw new ArgumentException("yExpect and yOutput must have the same dimensions.");
        }
        
        Matrix<double> differenceEnhanced = (yExpect - yOutput).Map(x => Math.Pow(x, 2));

        double sum = differenceEnhanced.Enumerate().Sum();
        double mse = sum / (differenceEnhanced.RowCount * differenceEnhanced.ColumnCount);

        double fitness = 1.0 / (1.0 + mse);
        return fitness;
    }

    public double MeanSquaredError(Matrix<double> yTrue, Matrix<double> yPred)
    {
        if (yTrue.RowCount != yPred.RowCount || yTrue.ColumnCount != yPred.ColumnCount)
        {
            throw new ArgumentException("yTrue and yPred must have the same dimensions.");
        }

        // Compute the element-wise difference
        var diff = yTrue - yPred;

        // Raise each element to the 4th power
        var diffPower4 = diff.PointwisePower(4);

        // Calculate the mean of all elements
        double sum = 0.0;
        int totalElements = diffPower4.RowCount * diffPower4.ColumnCount;

        foreach (var value in diffPower4.Enumerate())
        {
            sum += value;
        }

        return sum / totalElements;
    }


    public List<IBaseLayer> createChildLayers(DenseLayer parent1Layer, DenseLayer parent2Layer, double fitnessAverage, double mutationMultiplier) {
        double mutationRate = .6;
        List<double> mutationRange = new List<double> { -1, 1 };

        DenseLayer child1Layer = new DenseLayer(parent1Layer.weights.ColumnCount, parent1Layer.weights.RowCount);
        DenseLayer child2Layer = new DenseLayer(parent2Layer.weights.ColumnCount, parent2Layer.weights.RowCount);

        int splitPointWeights = randomStatic.Next(1, parent1Layer.weights.RowCount);
        if (splitPointWeights != 0 && splitPointWeights != parent1Layer.weights.RowCount) {
            (Matrix<double> parent1LeftWeights, Matrix<double> parent1RightWeights) = parent1Layer.splitWeights(splitPointWeights);
            (Matrix<double> parent2LeftWeights, Matrix<double> parent2RightWeights) = parent2Layer.splitWeights(splitPointWeights);

            child1Layer.weights = parent1LeftWeights.Stack(parent1RightWeights);
            child2Layer.weights = parent2LeftWeights.Stack(parent2RightWeights);
        }
        else {
            child1Layer.weights = parent1Layer.weights.Clone();
            child2Layer.weights = parent2Layer.weights.Clone();
        }

        int splitPointBias = randomStatic.Next(1, parent1Layer.bias.RowCount);
        if (splitPointBias != 0 && splitPointBias != parent1Layer.bias.RowCount) {
            (Matrix<double> parent1LeftBias, Matrix<double> parent1RightBias) = parent1Layer.splitBias(splitPointBias);
            (Matrix<double> parent2LeftBias, Matrix<double> parent2RightBias) = parent2Layer.splitBias(splitPointBias);

            child1Layer.bias = parent1LeftBias.Stack(parent1RightBias);
            child2Layer.bias = parent2LeftBias.Stack(parent2RightBias);
        }
        else {
            child1Layer.bias = parent1Layer.bias.Clone();
            child2Layer.bias = parent2Layer.bias.Clone();
        }

        child1Layer.mutateWeights(mutationRate, mutationRange, mutationMultiplier);
        child2Layer.mutateWeights(mutationRate, mutationRange, mutationMultiplier);

        child1Layer.mutateBias(mutationRate, mutationRange, mutationMultiplier);
        child2Layer.mutateBias(mutationRate, mutationRange, mutationMultiplier);

        return new List<IBaseLayer> { child1Layer, child2Layer };
    }

    private List<NeuralNetwork> eliteismSelection(List<NeuralNetwork> population, int percentElite = 10) {
        population = population
            .OrderByDescending(network => network.fitnessScore)
            .ToList();
        population = population.GetRange(0, population.Count / percentElite);
        return population;
    }

    private List<NeuralNetwork> eliteismDiversitySelection(List<NeuralNetwork> population, int percentElite = 8, int percentDiversity = 2) {
        // Sort the population in descending order based on fitness scores
        population = population
            .OrderByDescending(network => network.fitnessScore)
            .ToList();

        int populationSize = population.Count;
        int eliteCount = (int)Math.Ceiling(populationSize * ((double)percentElite / 100)); // Top 8% of the population
        int diversityCount = (int)Math.Ceiling(populationSize * ((double)percentDiversity / 100)); // Bottom 2% of the population

        if (eliteCount + diversityCount > populationSize) {
            throw new InvalidOperationException("Elite and diversity count exceeds population size.");
        }

        // Select top 8% as elites
        List<NeuralNetwork> elites = population
            .Take(eliteCount)
            .ToList();

        // Select bottom 2% for diversity
        List<NeuralNetwork> diversityIndividuals = population
            .Skip(populationSize - diversityCount)
            .Take(diversityCount)
            .ToList();

        elites.AddRange(diversityIndividuals);
        population = elites;
        return population;
    }

    private List<NeuralNetwork> populationBottomMutation(List<NeuralNetwork> population, double mutationRate, List<double> mutationRange, 
        double mutationMultiplier, int bottomMutationPercent) 
    {
        int populationSize = population.Count;
        int keepCount = (int)Math.Ceiling(populationSize * ((double)(100 - bottomMutationPercent) / 100)); // Corrected calculation
        int mutateCount = (int)Math.Ceiling(populationSize * ((double)(bottomMutationPercent) / 100)); // Corrected calculation

        if (keepCount + mutateCount > populationSize) {
            throw new InvalidOperationException("Elite and diversity count exceeds population size.");
        }

        // Select top (100 - bottomMutationPercent)% as kept individuals
        List<NeuralNetwork> keepsIndividuals = population
            .Take(keepCount)
            .ToList();

        // Select bottom bottomMutationPercent% for mutation
        List<NeuralNetwork> diversityIndividuals = population
            .Skip(populationSize - mutateCount)
            .Take(mutateCount)
            .ToList();
            
        // Mutate the selected diversity individuals
        foreach (NeuralNetwork network in diversityIndividuals) {
            foreach (IBaseLayer layer in network.layers) {
                if (layer is DenseLayer denseLayer) {
                    denseLayer.mutateWeights(mutationRate, mutationRange, mutationMultiplier);
                    denseLayer.mutateBias(mutationRate, mutationRange, mutationMultiplier);
                }
            }
        }

        // Combine kept and mutated individuals to form the new population
        keepsIndividuals.AddRange(diversityIndividuals);

        return keepsIndividuals;
    }

    public List<NeuralNetwork> trainGenetically(List<NeuralNetwork> population, 
        Matrix<double> xTrain, Matrix<double> yTrain, int generationlimit = 100) {
            bool earlyBreak = false;
            int count = 1;
            double mutationMultiplier = 1.0;
            double last100Avg = 0.0;
            double current100Avg = 0.0;
            for (int genIndex = 0; genIndex < generationlimit; genIndex++) {
                // Console.WriteLine($"{DateTime.Now}: Generation {genIndex} Started.");
                //Reset fitness scores
                if (genIndex % 100 == 0) {
                    Console.WriteLine($"Last 100 average: {last100Avg:0.0000}");
                    Console.WriteLine($"Current 100 average: {current100Avg:0.0000}");
                    if (last100Avg-current100Avg > -.1) {
                        mutationMultiplier+=0.25;
                        mutationMultiplier.Round(2);
                    }
                    else {
                        mutationMultiplier = 1;
                    }
                    Console.WriteLine($"Mutation multiplier: {mutationMultiplier}");
                    last100Avg = current100Avg;
                }
                foreach (NeuralNetwork network in population) {
                    network.fitnessScore = 0;
                }

                // Console.WriteLine($"{DateTime.Now} Networks set to 0 fitness score.");
                for (int i = 0; i < xTrain.RowCount/2; i++) { //9 is an arbitrary number of examples to run for each generation
                    Parallel.ForEach(population, network =>
                    {
                        Matrix<double> output = network.predictOutcome(xTrain.SubMatrix(i % xTrain.RowCount, 1, 0, xTrain.ColumnCount));
                        network.fitnessScore += ErrorBasedFitness(yTrain.SubMatrix(i % yTrain.RowCount, 1, 0, yTrain.ColumnCount), output);
                    });
                }

                // Console.WriteLine($"{DateTime.Now}: Fitness calculated for all.");

                population = eliteismDiversitySelection(population, 8, 2);

                double populationAverage = population.Average(network => network.fitnessScore);

                Console.WriteLine($"Generation {genIndex + 1} complete. ");
                Console.WriteLine($"Top fitness score: {population[0].fitnessScore:0.0000}");
                Console.WriteLine($"Average fitness score: {populationAverage:0.0000}");
                current100Avg = (populationAverage+current100Avg)/2;

                // if (population[0].fitnessScore > 8.997) {
                //     earlyBreak = true;
                // }
                // Console.WriteLine($"{DateTime.Now}: Top 10% done.");

                List<NeuralNetwork> newPopulation = new List<NeuralNetwork>();

                // Console.WriteLine($"{DateTime.Now}: Start population children.");

                // Suppose we have these outside the loop:
                
                int initialPopulationCount = population.Count;
                int totalOffspringNeeded = initialPopulationCount * 9; // The desired multiplication factor
                // Create a thread-local Random
                ThreadLocal<Random> threadLocalRandom = new(() => new Random(Guid.NewGuid().GetHashCode()));

                // We can use a concurrent data structure or thread-local lists
                var offspringBag = new ConcurrentBag<NeuralNetwork>();

                // Run in parallel:
                Parallel.For(0, totalOffspringNeeded / 2, (i) =>
                {
                    // Each iteration produces 2 children
                    var rand = threadLocalRandom.Value;  // Safe to use now
                    NeuralNetwork parent1 = population[rand!.Next(population.Count)];
                    NeuralNetwork parent2 = population[rand.Next(population.Count)];

                    List<IBaseLayer> child1Layers = new List<IBaseLayer>();
                    List<IBaseLayer> child2Layers = new List<IBaseLayer>();

                    double averageFitness = (parent1.fitnessScore + parent2.fitnessScore) / 2.0;

                    for (int layerIndex = 0; layerIndex < parent1.layers.Count; layerIndex++)
                    {
                        if (parent1.layers[layerIndex] is not DenseLayer)
                            continue;

                        List<IBaseLayer> childrenLayers = createChildLayers(
                            (DenseLayer)parent1.layers[layerIndex],
                            (DenseLayer)parent2.layers[layerIndex],
                            averageFitness,
                            mutationMultiplier
                        );

                        IBaseLayer child1Layer = childrenLayers[0];
                        IBaseLayer child2Layer = childrenLayers[1];

                        child1Layers.Add(child1Layer);
                        child2Layers.Add(child2Layer);

                        // Add the activation after each dense layer
                        child1Layers.Add(new ActivationTanh());
                        child2Layers.Add(new ActivationTanh());
                    }

                    NeuralNetwork child1 = new NeuralNetwork(child1Layers);
                    NeuralNetwork child2 = new NeuralNetwork(child2Layers);

                    // Add children to the concurrent collection
                    offspringBag.Add(child1);
                    offspringBag.Add(child2);
                });

                // After parallel execution, combine offspring with population
                newPopulation = offspringBag.ToList();
                newPopulation.AddRange(population);
                //populationBottomMutation(population, 0.5, new List<double> { -1, 1 }, mutationMultiplier, 20)
                population = newPopulation;
                // Console.WriteLine($"{DateTime.Now}: End population restart.");
                if (earlyBreak) {
                    break;
                }
                if (genIndex == generationlimit - 1) {
                    Console.WriteLine("Generation limit reached.");
                    Console.WriteLine("Enter additional generations to run: ");
                    generationlimit+= Convert.ToInt32(Console.ReadLine());
                }
            }
            return population;
        }

    public List<NeuralNetwork> PerformWeightedSampling(List<NeuralNetwork> population, int sampleCount)
    {
        // Calculate the total fitness score
        double totalFitness = population.Sum(network => network.fitnessScore);

        if (totalFitness == 0)
        {
            throw new InvalidOperationException("Total fitness is zero. Cannot perform weighted sampling.");
        }

        // Normalize fitness scores to get probabilities
        double[] probabilities = population
            .Select(network => network.fitnessScore / totalFitness)
            .ToArray();

        // Build the cumulative distribution function (CDF)
        double[] cdf = new double[probabilities.Length];
        cdf[0] = probabilities[0];
        for (int i = 1; i < probabilities.Length; i++)
        {
            cdf[i] = cdf[i - 1] + probabilities[i];
        }

        // Perform weighted random sampling
        List<NeuralNetwork> selected = new List<NeuralNetwork>();
        for (int i = 0; i < sampleCount; i++)
        {
            double rand = randomStatic.NextDouble(); // Random number between 0 and 1
            int index = Array.BinarySearch(cdf, rand);
            if (index < 0)
            {
                index = ~index; // Find the first element larger than rand
            }

            selected.Add(population[index]);
        }

        return selected;
    }
}