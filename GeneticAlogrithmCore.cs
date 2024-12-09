using System.Collections.Concurrent;
using System.Net.NetworkInformation;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

public class GeneticAlgorithmCore{
    public GeneticAlgorithmCore() {
    }
    private static readonly Random randomStatic = new Random();
    public double mutationMultiplier = 1.0;
    public List<double> mutationRange = new List<double> { -1, 1 };

    public double mutationRate = 0.6;
    public double last100Avg = 0.0;
    public double current100Avg = 0.0;
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

    public static double CalendarBasedFitness(Matrix<double> input, Matrix<double> output) {
        double fitness = 0;
        if (input.RowCount != output.RowCount || input.ColumnCount != output.ColumnCount) {
            throw new ArgumentException("Input and output must have the same dimensions.");
        }
        for (int i = 0; i < input.RowCount; i++) {
            for (int j = 0; j < input.ColumnCount; j++) {
                switch (input[i, j]) {
                    case 1:
                        fitness += output[i, j] > 1 ? 0 : output[i, j];
                        break;
                    case 0:
                        fitness += 1 - Math.Abs(output[i, j]);
                        break;
                    default:
                        throw new ArgumentException("Invalid input value.");
                }
            }
        }
        return fitness;
    }
    public List<IBaseLayer> createChildLayers(DenseLayer parent1Layer, DenseLayer parent2Layer, double fitnessAverage, double mutationMultiplier) {

        mutationRate = (96*400 - fitnessAverage) / 96*400; //500 is the size of the input dataset

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

    private void resetFitnessScores(List<NeuralNetwork> population) {
        foreach (NeuralNetwork network in population) {
            network.fitnessScore = 0;
        }
    }

    private void calculatePopulationFitness(List<NeuralNetwork> population, Matrix<double> xTrain) {
        for (int i = 0; i < xTrain.RowCount; i++) {
            Parallel.ForEach(population, network =>
            {
                Matrix<double> output = network.predictOutcome(xTrain.SubMatrix(i % xTrain.RowCount, 1, 0, xTrain.ColumnCount));
                network.fitnessScore += CalendarBasedFitness(xTrain.SubMatrix(i % xTrain.RowCount, 1, 0, xTrain.ColumnCount), output);
            });
            // foreach (var network in population)
            // {
            //     Matrix<double> inputSubMatrix = xTrain.SubMatrix(i % xTrain.RowCount, 1, 0, xTrain.ColumnCount);
            //     Matrix<double> output = network.predictOutcome(inputSubMatrix);
            //     double fitness = CalendarBasedFitness(inputSubMatrix, output);
            //     network.fitnessScore += fitness;
            // }
        }
    }

    private void handleGenerationProgress(int genIndex) {
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
    }
    
    private List<NeuralNetwork> repopulatePopulation(List<NeuralNetwork> population, int migrants = 5, int clones = 5) {
        List<NeuralNetwork> newPopulation = new List<NeuralNetwork>();
        // Suppose we have these outside the loop:
        
        int initialPopulationCount = population.Count;
        int totalOffspringNeeded = initialPopulationCount * 9 - migrants - clones; // The desired multiplication factor
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
        newPopulation.AddRange(createTravelers(migrants));
        newPopulation.AddRange(createClones(clones, population[0]));
        newPopulation.AddRange(populationBottomMutation(population, mutationRate, mutationRange, mutationMultiplier, 90));
        return newPopulation;
    }

    public List<NeuralNetwork> createTravelers(int travelers) {
        List<NeuralNetwork> travelerPopulation = new List<NeuralNetwork>();
        for (int index = 0; index < travelers; index++) {
            NeuralNetwork network = new NeuralNetwork(
                new List<IBaseLayer>{
                    new DenseLayer(96, 124),
                    new ActivationTanh(),
                    new DenseLayer(124, 96),
                    new ActivationSigmoid(),
                    new DenseLayer(96, 124),
                    new ActivationTanh(),
                    new DenseLayer(124, 96),
                    new ActivationSigmoid()
                }
            );
            travelerPopulation.Add(network);
        }
        return travelerPopulation;
    }

    public List<NeuralNetwork> createClones(int clones, NeuralNetwork topPerformer) {
        List<NeuralNetwork> clonePopulation = new List<NeuralNetwork>();
        for (int index = 0; index < clones; index++) {
            List<IBaseLayer> cloneLayers = new List<IBaseLayer>();
            foreach (IBaseLayer layer in topPerformer.layers) {
                cloneLayers.Add(layer.cloneLayer());
            }
            NeuralNetwork clone = new NeuralNetwork(cloneLayers);
            clonePopulation.Add(clone);
        }
        return clonePopulation;
    }
    public List<NeuralNetwork> trainGenetically(List<NeuralNetwork> population, 
        Matrix<double> xTrain, int generationlimit = 100) {
            bool earlyBreak = false;
            for (int genIndex = 0; genIndex < generationlimit; genIndex++) {
                Console.WriteLine($"Population Size: {population.Count}");
                handleGenerationProgress(genIndex);

                resetFitnessScores(population);

                calculatePopulationFitness(population, xTrain);

                population = eliteismDiversitySelection(population, 4, 6); //Must add to 10

                double populationAverage = population.Average(network => network.fitnessScore);

                Console.WriteLine($"Generation {genIndex + 1} complete. ");
                Console.WriteLine($"Top fitness score: {population[0].fitnessScore:0.0000}");
                Console.WriteLine($"Average fitness score: {populationAverage:0.0000}");
                current100Avg = (populationAverage+current100Avg)/2;

                // if (population[0].fitnessScore > 8.997) {
                //     earlyBreak = true;
                // }
                
                population = repopulatePopulation(population, migrants: 10, clones: 10);
                // Console.WriteLine($"{DateTime.Now}: End population restart.");
                if (earlyBreak) {
                    break;
                }
                if (genIndex == generationlimit - 1) {
                    Console.WriteLine("Generation limit reached.");
                    Console.WriteLine("Enter additional generations to run: ");
                    string inputedValue = Console.ReadLine();
                    generationlimit+= Convert.ToInt32(inputedValue == String.Empty ? "0" : inputedValue);
                }
            }
            population = population
                .OrderByDescending(network => network.fitnessScore)
                .ToList();
            return population;
        }
}