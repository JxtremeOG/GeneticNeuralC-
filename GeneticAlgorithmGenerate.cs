using System.Collections;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;

public static class ThreadSafeRandom
{
    private static int seed = Environment.TickCount;

    private static ThreadLocal<Random> threadLocalRandom = new ThreadLocal<Random>(
        () => new Random(Interlocked.Increment(ref seed))
    );

    public static Random Instance => threadLocalRandom.Value;
}
public class ScheduleBitMap {
    private static readonly ThreadLocal<Random> threadLocalRandom = new ThreadLocal<Random>(() =>
        new Random(ThreadSafeRandom.Instance.Next()));
    private BitArray schedule;
    public int scheduleSize;
    public double scheduleDeviation;
    private BitArray scheduleBase;
    public double fitness;
    public HashSet<int> taskIndexes = new HashSet<int>();
    public List<int> vacantSegments;
    public ScheduleBitMap(BitArray schedulePassed, List<int> vacantSegmentsPassed) {
        schedule = schedulePassed;
        scheduleBase = schedulePassed;
        scheduleSize = schedulePassed.Length;
        vacantSegments = vacantSegmentsPassed;
        fitness = 0;
    }

    public ScheduleBitMap(ScheduleBitMap other)
    {
        // Deep copy of BitArray
        schedule = new BitArray(other.schedule);
        scheduleBase = new BitArray(other.scheduleBase);
        
        // Copy primitive types
        scheduleSize = other.scheduleSize;
        scheduleDeviation = other.scheduleDeviation;
        fitness = other.fitness;
        
        // Deep copy of taskIndexes (ImmutableHashSet ensures thread safety)
        taskIndexes = other.taskIndexes;
        
        // Deep copy of vacantSegments
        vacantSegments = new List<int>(other.vacantSegments);
    }
    public void addTask(int taskSize)
    {
        int validFlips = 0;
        while (validFlips < taskSize)
        {
            int flipIndex = threadLocalRandom.Value.Next(0, schedule.Length);
            if (!schedule[flipIndex])
            {
                schedule[flipIndex] = true;
                taskIndexes.Add(flipIndex);
                validFlips++;
            }
        }
    }

    public BitArray getSchedule() {
        return schedule;
    }

    public BitArray getBaseSchedule() {
        return scheduleBase;
    }

    public bool getBitValue(int bitIndex) {
        return schedule[bitIndex];
    }

    public bool getBaseBitValue(int bitIndex) {
        return scheduleBase[bitIndex];
    }

    public bool mutateBit(int bitIndex) {
        if (schedule[bitIndex] == true) {
            // if (!taskIndexs.Contains(bitIndex))
            //     return false;
            schedule[bitIndex] = false;
            taskIndexes.Remove(bitIndex);
        }
        else {
            schedule[bitIndex] = true;
            taskIndexes.Add(bitIndex);
        }
        return true;
    }
}

public class GeneticAlgorithmGenerate {
    public int populationSize;
    public int taskSize;
    public static Random random = new Random();
    public int scheduleSize;
    public int mutationChance;
    public Stopwatch geneticStopWatch = Stopwatch.StartNew();
    public ScheduleBitMap overallBestSchedule;
    public int immigrantCount;
    public int generationCount;
    public double previousBestFitness = 0;
    public int generationsWithoutImprovement = 0;
    public double baseFitness = 0;
    public BitArray scheduleBase;

    public ConcurrentDictionary<int, double> segmentScores = new ConcurrentDictionary<int, double>();

    public List<int> vacantSegments = new List<int>();
    public FitnessCore fitnessCore = new FitnessCore();
    public List<ScheduleBitMap> population = new List<ScheduleBitMap>();
    public ScheduleBitMap singleTaskManager;
    public Random geneticRandom = new Random();
    private object[] locks;
    public GeneticAlgorithmGenerate(int scheduleSizePassed, int taskSizePassed, 
    int populationSizePassed, int mutationChancePassed, int generationSizePassed, 
    int immigrantCountPassed) {
        populationSize = populationSizePassed;
        taskSize = taskSizePassed;
        scheduleSize = scheduleSizePassed;
        mutationChance = mutationChancePassed;
        generationCount = generationSizePassed;
        immigrantCount = immigrantCountPassed;
        scheduleBase = GenerateOrganizedSchedule();

        for (int segmentIndex = 0; segmentIndex < scheduleSize; segmentIndex++) { //Creates a dictionary of all open segments in base
            if (!scheduleBase[segmentIndex]) {
                vacantSegments.Add(segmentIndex);
                segmentScores.TryAdd(segmentIndex, 0);
            }
        }

        locks = new object[scheduleSizePassed];
        for (int i = 0; i < locks.Length; i++)
            locks[i] = new object();

        fitnessCore.PreCalculateClumpScores(scheduleSize);
        fitnessCore.PreCalculateTaskClumpScores(taskSize);
        singleTaskManager = new ScheduleBitMap(new BitArray(scheduleBase), vacantSegments);
        baseFitness = fitnessCore.FitnessFunction(singleTaskManager);
    }
    public void loneSegmentScores(List<int> vacantSegments) {
        foreach (int segment in vacantSegments) {
            singleTaskManager.mutateBit(segment);
            singleTaskManager.taskIndexes.Add(segment);
            fitnessCore.FitnessFunction(singleTaskManager);
            segmentScores[segment] += singleTaskManager.fitness - baseFitness;
            singleTaskManager.mutateBit(segment);
            singleTaskManager.taskIndexes.Remove(segment);
        }
    }
    public void MinMaxNormalizeSegmentScores()
    {
        double minScore = segmentScores.Values.Min();
        double maxScore = segmentScores.Values.Max();
        
        double range = maxScore - minScore;
        if (range <= 0)
        {
            throw new Exception("Range of segment scores is 0 or negative. Cannot normalize.");
        }
        
        // Rescale values to the [0, 1] range
        foreach (int key in segmentScores.Keys.ToList())
        {
            segmentScores[key] = Math.Round((segmentScores[key] - minScore) / range, 2);
        }
    }

    public BitArray GenerateOrganizedSchedule() {
        BitArray schedule = new BitArray(scheduleSize);
        int index = 0;
        while (index < scheduleSize) {
            if (index % 96 == 0) { // Generate sleep time
                for (int i = 0; i < 32; i++) {
                    schedule[index + i] = true;
                }
                index+=32;
            }
            else if (index % 96 == 76) { // Generate dinner time
                int dinnerLength = random.Next(3,6);
                for (int i = 0; i < dinnerLength; i++) {
                    schedule[index + i] = true;
                }
                index+=dinnerLength;
            }
            else
                index+=1;
        }

        for (int i = 0; i < scheduleSize; i++) {
            if (!schedule[i])
                schedule[i] = geneticRandom.Next(0, 2) == 1;
        }
        scheduleBase = schedule;
        return schedule;
    }
    public void generateFreshPopulation() {
        population = new List<ScheduleBitMap>();
        for (int i = 0; i < populationSize; i++) {
            ScheduleBitMap schedule = new ScheduleBitMap(new BitArray(scheduleBase), vacantSegments);
            schedule.addTask(taskSize);
            population.Add(schedule);
        }
    }

    public Task UpdateSegmentScoresAsync(List<ScheduleBitMap> population)
    {
        return Task.Run(() =>
        {
            // Use Parallel.ForEach with thread-local dictionaries
            Parallel.ForEach(population,
                () => new Dictionary<int, double>(), // Initialize thread-local dictionary
                (schedule, loopState, localDict) =>
                {
                    double newFitnessValue = schedule.fitness - baseFitness;
                    foreach (int taskIndex in schedule.taskIndexes)
                    {
                        if (localDict.TryGetValue(taskIndex, out double existing))
                        {
                            if (newFitnessValue > existing)
                            {
                                localDict[taskIndex] = newFitnessValue;
                            }
                        }
                        else
                        {
                            localDict[taskIndex] = newFitnessValue;
                        }
                    }
                    return localDict;
                },
                localDict =>
                {
                    // Merge thread-local dictionaries into the global ConcurrentDictionary
                    foreach (var kvp in localDict)
                    {
                        segmentScores.AddOrUpdate(
                            kvp.Key,
                            kvp.Value,
                            (key, existingValue) => Math.Max(existingValue, kvp.Value)
                        );
                    }
                }
            );
        });
    }


    public async Task<ScheduleBitMap> TrainGenetically() {
        geneticStopWatch.Start();
        ScheduleMutator scheduleMutator = new ScheduleMutator(mutationChance);
        int generationsWithoutImprovementLimit = 50;
        int generationsWithoutImprovement = 0;
        double previousBestFitness = 0;
        Console.WriteLine($"Training with task size: {taskSize} and population size: {populationSize} and generation count: {generationCount}");
        generateFreshPopulation();
        overallBestSchedule = population[0];
        for (int i = 0; i < generationCount; i++) {
            // Console.WriteLine($"Generation: {i} and population size: {population.Count}");
            ParallelOptions parallelOptions = new ParallelOptions{ MaxDegreeOfParallelism = 16  /* limit to 4 concurrent threads */ };

            Parallel.ForEach(population, parallelOptions, schedule =>
            {
                schedule.fitness = 0;
                fitnessCore.FitnessFunction(schedule);
            });

            List<ScheduleBitMap> populationSnapshot = population.Select(s => new ScheduleBitMap(s)).ToList();
            Task updateTask = UpdateSegmentScoresAsync(populationSnapshot);

            // foreach (ScheduleBitMap schedule in population) {
            //     schedule.fitness = 0;
            //     fitnessCore.FitnessFunction(schedule);
            // }
            int remianingPopulation = (int)(populationSize * .3); //.1 = 10% of the population
            List<ScheduleBitMap> newPopulation = new List<ScheduleBitMap>();
            newPopulation = population.OrderByDescending(x => x.fitness).Take(remianingPopulation).ToList();

            double currentBestFitness = newPopulation[0].fitness;
            if (currentBestFitness > overallBestSchedule.fitness) {
                overallBestSchedule = newPopulation[0];
            }

            if (i % 10 == 0) {
                // Console.WriteLine($"Generation: {i} Top performer fitness: {currentBestFitness} Overall best fitness: {overallBestSchedule.fitness}");
                newPopulation.Add(scheduleMutator.MutateShiftMode(newPopulation[0], scheduleMutator.getTaskClumps(newPopulation[0])));
            }

            if (i % 50 == 0)
                Console.WriteLine($"Generation: {i} Overall best fitness: {overallBestSchedule.fitness}");

            if (previousBestFitness >= currentBestFitness) {
                generationsWithoutImprovement++;
            }
            else {
                generationsWithoutImprovement = 0;
            }
            previousBestFitness = currentBestFitness;

            if (generationsWithoutImprovement > generationsWithoutImprovementLimit) {
                // Console.WriteLine($"No improvement for {generationsWithoutImprovementLimit} generations. Generating fresh population");
                generateFreshPopulation();
                generationsWithoutImprovement = 0;
            }
            else {
                newPopulation.AddRange(population
                    .OrderBy(x => x.fitness)
                    .Skip(immigrantCount)
                    .Take(immigrantCount * 3)
                    .ToList());
                for (int j = 0; j < immigrantCount-1; j++) {
                    ScheduleBitMap addedSchedule = new ScheduleBitMap(new BitArray(scheduleBase), vacantSegments);
                    addedSchedule.addTask(taskSize);
                    newPopulation.Add(addedSchedule);
                }
                
                int remaining = populationSize - newPopulation.Count;
                int batchSize = 50; // Adjust based on performance tests

                var parentPairs = new List<Tuple<ScheduleBitMap, ScheduleBitMap>>();
                for (int j = 0; j < remaining / 2; j++)
                {
                    ScheduleBitMap parent1 = population[ThreadSafeRandom.Instance.Next(0, population.Count)];
                    ScheduleBitMap parent2 = population[ThreadSafeRandom.Instance.Next(0, population.Count)];
                    parentPairs.Add(new Tuple<ScheduleBitMap, ScheduleBitMap>(parent1, parent2));
                }

                var childBag = new ConcurrentBag<ScheduleBitMap>();

                Parallel.ForEach(parentPairs, new ParallelOptions { MaxDegreeOfParallelism = 16 }, pair =>
                {
                    var children = CrossOver(pair.Item1, pair.Item2);
                    var child1 = scheduleMutator.Mutate(children.Item1);
                    var child2 = scheduleMutator.Mutate(children.Item2);
                    childBag.Add(child1);
                    childBag.Add(child2);
                });

                // Add children to newPopulation
                foreach (var child in childBag)
                {
                    if (newPopulation.Count >= populationSize)
                        break;
                    newPopulation.Add(child);
                }

                // Handle odd population sizes (if batchSize is too big)
                while (newPopulation.Count < populationSize)
                {
                    ScheduleBitMap parent1 = population[ThreadSafeRandom.Instance.Next(0, population.Count)];
                    ScheduleBitMap parent2 = population[ThreadSafeRandom.Instance.Next(0, population.Count)];
                    var children = CrossOver(parent1, parent2);
                    var child1 = scheduleMutator.Mutate(children.Item1);
                    newPopulation.Add(child1);
                    if (newPopulation.Count < populationSize)
                    {
                        var child2 = scheduleMutator.Mutate(children.Item2);
                        newPopulation.Add(child2);
                    }
                }
                population = newPopulation;
                await updateTask;
            }
        }
        Console.WriteLine(geneticStopWatch.Elapsed);
        loneSegmentScores(vacantSegments);
        Console.WriteLine(geneticStopWatch.Elapsed);
        MinMaxNormalizeSegmentScores();
        Console.WriteLine(geneticStopWatch.Elapsed);
        geneticStopWatch.Stop();
        return overallBestSchedule;
    }

    public Tuple<ScheduleBitMap, ScheduleBitMap> CrossOver(ScheduleBitMap schedule1, ScheduleBitMap schedule2) {
        ScheduleBitMap childSchedule1 = new ScheduleBitMap(new BitArray(scheduleBase), vacantSegments);
        ScheduleBitMap childSchedule2 = new ScheduleBitMap(new BitArray(scheduleBase), vacantSegments);
        List<int> child1TaskIndex = new List<int>();
        List<int> child2TaskIndex = new List<int>();
        List<int> combinedParentTasks = new List<int>();
        combinedParentTasks.AddRange(schedule1.taskIndexes);
        combinedParentTasks.AddRange(schedule2.taskIndexes);

        int halfCount = (int)(combinedParentTasks.Count / 2);
        while (child1TaskIndex.Count != child1TaskIndex.Distinct().ToList().Count || child1TaskIndex.Count == 0) {
            Shuffle(combinedParentTasks, geneticRandom); 
            child1TaskIndex = combinedParentTasks.Take(halfCount).ToList();
        }
        while (child2TaskIndex.Count != child2TaskIndex.Distinct().ToList().Count || child2TaskIndex.Count == 0) {
            Shuffle(combinedParentTasks, geneticRandom); 
            child2TaskIndex = combinedParentTasks.Take(halfCount).ToList();
        }
        for (int i = 0; i < taskSize; i++) {
            childSchedule1.mutateBit(child1TaskIndex[i]);
            childSchedule2.mutateBit(child2TaskIndex[i]);
        }

        // Console.WriteLine($"Child 1: {childSchedule1.taskIndexs.Count} Child 2: {childSchedule2.taskIndexs.Count}");
        return new Tuple<ScheduleBitMap, ScheduleBitMap>(childSchedule1, childSchedule2);
    }
    public void Shuffle<T>(IList<T> list, Random rng) {
    for (int i = list.Count - 1; i > 0; i--) {
        int j = rng.Next(i + 1);
        (list[i], list[j]) = (list[j], list[i]);
    }
}
}