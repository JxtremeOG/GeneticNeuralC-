using System.Collections;
using System.Formats.Asn1;
using Tensorflow;

public class ShceduleBitMap {
    public static Random random = new Random();
    private BitArray schedule;
    private BitArray scheduleBase;
    public int fitness;
    public HashSet<int> taskIndexs = new HashSet<int>();
    public ShceduleBitMap(BitArray schedulePassed) {
        schedule = schedulePassed;
        scheduleBase = schedulePassed;
        fitness = 0;
    }
    public void addTask(int taskSize) {
        int validFlips = 0;
        while (validFlips < taskSize) {
            int flipIndex = random.Next(0, schedule.Length);
            if (schedule[flipIndex] == false) {
                schedule[flipIndex] = true;
                taskIndexs.Add(flipIndex);
                validFlips++;
            }
        }
    }

    public BitArray getSchedule() {
        return schedule;
    }

    public bool getBitValue(int bitIndex) {
        return schedule[bitIndex];
    }

    public bool mutateBit(int bitIndex) {
        if (schedule[bitIndex] == true) {
            // if (!taskIndexs.Contains(bitIndex))
            //     return false;
            schedule[bitIndex] = false;
            taskIndexs.Remove(bitIndex);
        }
        else {
            schedule[bitIndex] = true;
            taskIndexs.Add(bitIndex);
        }
        return true;
    }
}

public class GeneticAlgorithmGenerate {
    public int populationSize;
    public int taskSize;
    public int scheduleSize;
    public int mutationChance;

    public int immigrantCount = 5;
    public int generationCount;
    public BitArray scheduleBase;
    public List<ShceduleBitMap> population = new List<ShceduleBitMap>();
    public Random geneticRandom = new Random();
    public GeneticAlgorithmGenerate(int scheduleSizePassed, int taskSizePassed, int populationSizePassed, int mutationChancePassed, int generationSizePassed) {
        populationSize = populationSizePassed;
        taskSize = taskSizePassed;
        scheduleSize = scheduleSizePassed;
        mutationChance = mutationChancePassed;
        generationCount = generationSizePassed;
        scheduleBase = GenerateSchedule();
    }
    public BitArray GenerateSchedule() {
        BitArray schedule = new BitArray(scheduleSize);
        for (int i = 0; i < scheduleSize; i++) {
            schedule[i] = geneticRandom.Next(0, 2) == 1;
        }
        scheduleBase = schedule;
        return schedule;
    }
    public ShceduleBitMap TournamentSelection(int tournamentSize) {
        int actualTournamentSize = Math.Min(tournamentSize, population.Count);
        List<ShceduleBitMap> tournament = new List<ShceduleBitMap>();
        // Randomly pick individuals to enter the tournament.
        for (int i = 0; i < actualTournamentSize; i++) {
            int randomIndex = geneticRandom.Next(0, population.Count);
            tournament.Add(population[randomIndex]);
        }
        // Sort them by fitness (descending)
        ShceduleBitMap winner = tournament.OrderByDescending(ind => ind.fitness).First();
        return winner;
    }
    public ShceduleBitMap TrainGenetically() {
        for (int i = 0; i < populationSize; i++) {
            ShceduleBitMap schedule = new ShceduleBitMap(new BitArray(scheduleBase));
            schedule.addTask(taskSize);
            population.Add(schedule);
        }
        for (int i = 0; i < generationCount; i++) {
            foreach (ShceduleBitMap schedule in population) {
                schedule.fitness = 0;
                FitnessFunction(schedule);
            }
            int remianingPopulation = (int)(populationSize * .3); //.1 = 10% of the population
            population = population.OrderByDescending(x => x.fitness).Take(remianingPopulation).ToList();
            List<ShceduleBitMap> newPopulation = new List<ShceduleBitMap>();

            if (i % 10 == 0)
                Console.WriteLine($"Generation: {i} Top performer fitness: {population[0].fitness}");

            newPopulation.AddRange(population);
            
            while (newPopulation.Count < populationSize - immigrantCount) {
                ShceduleBitMap parent1 = TournamentSelection(5); // sample size = 5, adjust as needed
                ShceduleBitMap parent2 = TournamentSelection(5);

                Tuple<ShceduleBitMap, ShceduleBitMap> children = CrossOver(parent1, parent2);
                ShceduleBitMap child1 = Mutate(children.Item1);
                ShceduleBitMap child2 = Mutate(children.Item2);

                newPopulation.Add(child1);
                if (newPopulation.Count < populationSize) {
                    newPopulation.Add(child2);
                }
            }
            for (int j = 0; j < immigrantCount; j++) {
                ShceduleBitMap addedSchedule = new ShceduleBitMap(new BitArray(scheduleBase));
                addedSchedule.addTask(taskSize);
                newPopulation.Add(addedSchedule);
            }
            population = newPopulation;
        }
        return population[0];
    }
    public double FitnessFunction(ShceduleBitMap schedule) {
        List<int> values = new List<int>{0,1,2,3,4,5,6,7,8,9,10,11};

        for (int i = 0; i < scheduleSize; i++) {
            if (schedule.getBitValue(i) && values.Contains(i)) {
                schedule.fitness++;
            }
        }
        return schedule.fitness;
    }

    public Tuple<ShceduleBitMap, ShceduleBitMap> CrossOver(ShceduleBitMap schedule1, ShceduleBitMap schedule2) {
        ShceduleBitMap childSchedule1 = new ShceduleBitMap(new BitArray(scheduleBase));
        ShceduleBitMap childSchedule2 = new ShceduleBitMap(new BitArray(scheduleBase));
        int child1FlipCount = 0;
        int child2FlipCount = 0;
        
        for (int i = 0; i < scheduleSize; i++) {
            if (schedule1.getBitValue(i) != scheduleBase[i]) {
                if (child1FlipCount >= taskSize) {
                    childSchedule2.mutateBit(i);
                }
                else if (child2FlipCount >= taskSize) {
                    childSchedule1.mutateBit(i);
                }
                else {
                    if (geneticRandom.Next(0, 2) == 1) {
                        childSchedule1.mutateBit(i);
                        child1FlipCount++;
                    }
                    else {
                        childSchedule2.mutateBit(i);
                        child2FlipCount++;
                    }
                }
            }
            else if (schedule2.getBitValue(i) != scheduleBase[i]) {
                if (child1FlipCount >= taskSize) {
                    childSchedule2.mutateBit(i);
                }
                else if (child2FlipCount >= taskSize) {
                    childSchedule1.mutateBit(i);
                }
                else {
                    if (geneticRandom.Next(0, 2) == 1) {
                        childSchedule1.mutateBit(i);
                        child1FlipCount++;
                    }
                    else {
                        childSchedule2.mutateBit(i);
                         child2FlipCount++;
                    }
                }
            }
        }
        return new Tuple<ShceduleBitMap, ShceduleBitMap>(childSchedule1, childSchedule2);
    }

    public ShceduleBitMap Mutate(ShceduleBitMap schedule) {
        var taskIndicesCopy = schedule.taskIndexs.ToList();
        foreach (int i in taskIndicesCopy) {
            if (geneticRandom.Next(0, 100) < mutationChance) {
                schedule.mutateBit(i);
                while (true) {
                    int randomIndex = geneticRandom.Next(0, scheduleSize);
                    if (!schedule.getBitValue(randomIndex)) {
                        schedule.mutateBit(randomIndex);
                        break;
                    }
                }
            }
        }
        return schedule;
    }
}