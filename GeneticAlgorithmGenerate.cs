using System.Collections;
using System.Formats.Asn1;
using System.Transactions;
using Tensorflow;

public class ScheduleBitMap {
    public static Random random = new Random();
    private BitArray schedule;
    public int scheduleSize;
    public double scheduleDeviation;
    private BitArray scheduleBase;
    public double fitness;
    public HashSet<int> taskIndexes = new HashSet<int>();
    public ScheduleBitMap(BitArray schedulePassed) {
        schedule = schedulePassed;
        scheduleBase = schedulePassed;
        scheduleSize = schedulePassed.Length;
        fitness = 0;
    }
    public void addTask(int taskSize) {
        int validFlips = 0;
        while (validFlips < taskSize) {
            int flipIndex = random.Next(0, schedule.Length);
            if (schedule[flipIndex] == false) {
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

    public ScheduleBitMap overallBestSchedule;
    public int immigrantCount;
    public int generationCount;
    public double previousBestFitness = 0;
    public int generationsWithoutImprovement = 0;
    public BitArray scheduleBase;
    public FitnessCore fitnessCore = new FitnessCore();
    public List<ScheduleBitMap> population = new List<ScheduleBitMap>();
    public Random geneticRandom = new Random();
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
    }
    public BitArray GenerateRandomSchedule() {
        BitArray schedule = new BitArray(scheduleSize);
        for (int i = 0; i < scheduleSize; i++) {
            schedule[i] = geneticRandom.Next(0, 2) == 1;
        }
        scheduleBase = schedule;
        return schedule;
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
            ScheduleBitMap schedule = new ScheduleBitMap(new BitArray(scheduleBase));
            schedule.addTask(taskSize);
            population.Add(schedule);
        }
    }
    public ScheduleBitMap TrainGenetically() {
        ScheduleMutator scheduleMutator = new ScheduleMutator(mutationChance);
        int generationsWithoutImprovementLimit = 50;
        int generationsWithoutImprovement = 0;
        double previousBestFitness = 0;
        Console.WriteLine($"Training with task size: {taskSize}");
        generateFreshPopulation();
        overallBestSchedule = population[0];
        fitnessCore.PreCalculateClumpScores(scheduleSize);
        for (int i = 0; i < generationCount; i++) {
            // Console.WriteLine($"Generation: {i} and population size: {population.Count}");
            foreach (ScheduleBitMap schedule in population) {
                schedule.fitness = 0;
                fitnessCore.FitnessFunction(schedule);
            }
            int remianingPopulation = (int)(populationSize * .3); //.1 = 10% of the population
            List<ScheduleBitMap> newPopulation = new List<ScheduleBitMap>();
            newPopulation = population.OrderByDescending(x => x.fitness).Take(remianingPopulation).ToList();

            double currentBestFitness = newPopulation[0].fitness;
            if (currentBestFitness > overallBestSchedule.fitness) {
                overallBestSchedule = newPopulation[0];
            }

            if (i % 10 == 0) {
                Console.WriteLine($"Generation: {i} Top performer fitness: {currentBestFitness} Overall best fitness: {overallBestSchedule.fitness}");
                newPopulation.Add(scheduleMutator.MutateShiftMode(newPopulation[0], scheduleMutator.getTaskClumps(newPopulation[0])));
            }

            if (previousBestFitness >= currentBestFitness) {
                generationsWithoutImprovement++;
            }
            else {
                generationsWithoutImprovement = 0;
            }
            previousBestFitness = currentBestFitness;

            if (generationsWithoutImprovement > generationsWithoutImprovementLimit) {
                Console.WriteLine($"No improvement for {generationsWithoutImprovementLimit} generations. Generating fresh population");
                generateFreshPopulation();
                generationsWithoutImprovement = 0;
            }
            else {
                newPopulation.AddRange(population
                    .OrderBy(x => x.fitness)
                    .Skip(immigrantCount)
                    .Take(immigrantCount * 3)
                    .ToList());
                for (int j = 0; j < immigrantCount; j++) {
                    ScheduleBitMap addedSchedule = new ScheduleBitMap(new BitArray(scheduleBase));
                    addedSchedule.addTask(taskSize);
                    newPopulation.Add(addedSchedule);
                }
                
                while (newPopulation.Count < populationSize) {
                    ScheduleBitMap parent1 = population[geneticRandom.Next(0, population.Count)];
                    ScheduleBitMap parent2 = population[geneticRandom.Next(0, population.Count)];

                    Tuple<ScheduleBitMap, ScheduleBitMap> children = CrossOver(parent1, parent2);
                    ScheduleBitMap child1 = scheduleMutator.Mutate(children.Item1);
                    ScheduleBitMap child2 = scheduleMutator.Mutate(children.Item2);

                    newPopulation.Add(child1);
                    if (newPopulation.Count < populationSize) {
                        newPopulation.Add(child2);
                    }
                }
                population = newPopulation;
            }
        }
        return overallBestSchedule;
    }

    public Tuple<ScheduleBitMap, ScheduleBitMap> CrossOver(ScheduleBitMap schedule1, ScheduleBitMap schedule2) {
        ScheduleBitMap childSchedule1 = new ScheduleBitMap(new BitArray(scheduleBase));
        ScheduleBitMap childSchedule2 = new ScheduleBitMap(new BitArray(scheduleBase));
        List<int> child1TaskIndex = new List<int>();
        List<int> child2TaskIndex = new List<int>();
        List<int> combinedParentTasks = new List<int>();
        combinedParentTasks.AddRange(schedule1.taskIndexes);
        combinedParentTasks.AddRange(schedule2.taskIndexes);

        int halfCount = (int)(combinedParentTasks.Count / 2);
        while (child1TaskIndex.Count != child1TaskIndex.Distinct().ToList().Count || child1TaskIndex.Count == 0) {
            child1TaskIndex = combinedParentTasks
                .OrderBy(_ => Guid.NewGuid())
                .Take(halfCount)              
                .ToList();
        }
        while (child2TaskIndex.Count != child2TaskIndex.Distinct().ToList().Count || child2TaskIndex.Count == 0) {
            child2TaskIndex = combinedParentTasks
                .OrderBy(_ => Guid.NewGuid())
                .Take(halfCount)              
                .ToList();
        }
        for (int i = 0; i < taskSize; i++) {
            childSchedule1.mutateBit(child1TaskIndex[i]);
            childSchedule2.mutateBit(child2TaskIndex[i]);
        }

        // Console.WriteLine($"Child 1: {childSchedule1.taskIndexs.Count} Child 2: {childSchedule2.taskIndexs.Count}");
        return new Tuple<ScheduleBitMap, ScheduleBitMap>(childSchedule1, childSchedule2);
    }
}