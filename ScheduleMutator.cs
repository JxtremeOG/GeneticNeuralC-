using MathNet.Numerics;
using Tensorflow;

public class ScheduleMutator {
    public static Random geneticRandom = new Random();
    public static int safeGuard = 20;
    public static int mutationChance;
    public ScheduleMutator(int MutationChance) {
        mutationChance = MutationChance;
    }
    public ScheduleBitMap MutateSingleBit(ScheduleBitMap schedule, int mutateIndex) {
        int iterationCount = 0;
        while (true && iterationCount < safeGuard) {
            int randomIndex = geneticRandom.Next(0, schedule.getSchedule().Length);
            if (!schedule.getBitValue(randomIndex)) {
                schedule.mutateBit(randomIndex);
                schedule.mutateBit(mutateIndex);
                break;
            }
            iterationCount++;
        }
        return schedule;
    }
    public ScheduleBitMap MutateSingleClump(ScheduleBitMap schedule, List<int> clumpIndexes) {
        int iterationCount = 0;
        while (iterationCount < safeGuard) {
            bool isValid = true;
            int randomIndex = geneticRandom.Next(0, schedule.getSchedule().Length - clumpIndexes.Count + 1);
            for (int i = 0; i < clumpIndexes.Count; i++) {
                if (schedule.getBitValue(randomIndex+i)) {
                    isValid = false;
                    break;
                }
            }
            if (isValid) {
                foreach (int i in clumpIndexes) {
                    schedule.mutateBit(i);
                }
                for (int i = 0; i < clumpIndexes.Count; i++) {
                    schedule.mutateBit(randomIndex+i);
                }
                break;
            }
            iterationCount++;
        }
        return schedule;
    }

    public ScheduleBitMap Mutate(ScheduleBitMap schedule) {
        List<List<int>> taskClumps = getTaskClumps(schedule);

        int mutationType = geneticRandom.Next(1, 5);
        switch (mutationType) {
            case 0: return MutateBitMode(schedule);            //Unused
            case 1: return MutateClumpMode(schedule, taskClumps);
            case 2: return MutateShiftMode(schedule, taskClumps);
            case 3: return MutateExactFitMode(schedule, taskClumps);
            case 4: return MutateCombineLoneSegments(schedule, taskClumps);
            default: throw new Exception("Invalid mutation type");
        }
    }

    public ScheduleBitMap MutateBitMode(ScheduleBitMap schedule) {
        //Single bit mutation
        var taskIndicesCopy = schedule.taskIndexes.ToList();
        taskIndicesCopy.Sort();
        foreach (int i in taskIndicesCopy) {
            if (geneticRandom.Next(0, 100) < mutationChance) {
                schedule = MutateSingleBit(schedule, i);
            }
        }
        return schedule;
    }
    public ScheduleBitMap MutateClumpMode(ScheduleBitMap schedule, List<List<int>> taskClumps) {
        //Task clump mutation
        
        foreach (List<int> clump in taskClumps) {
            if (geneticRandom.Next(0, 100) < mutationChance) {
                MutateSingleClump(schedule, clump);
            }
        }
        return schedule;
    }
    public ScheduleBitMap MutateShiftMode(ScheduleBitMap schedule, List<List<int>> taskClumps) {

        foreach (List<int> clump in taskClumps) {
            if (geneticRandom.Next(0, 100) < mutationChance) {
                if (geneticRandom.Next(0, 2) == 1) { //Left shift
                    int workingIndex = clump[0];
                    while (workingIndex > 0 && !schedule.getBitValue(workingIndex-1)) {
                        workingIndex--;
                    }
                    for (int i = 0; i < clump.Count; i++) {
                        schedule.mutateBit(clump[i]);
                        schedule.mutateBit(workingIndex+i);
                    }
                }
                else { //Right shift
                    int workingIndex = clump[clump.Count-1];
                    while (workingIndex < schedule.getSchedule().Length-1 &&!schedule.getBitValue(workingIndex+1)) {
                        workingIndex++;
                    }
                    for (int i = clump.Count-1; i >= 0; i--) {
                        schedule.mutateBit(clump[i]);
                        schedule.mutateBit(workingIndex - i);
                    }
                }
            }
        }
        return schedule;
    }
    public ScheduleBitMap MutateExactFitMode(ScheduleBitMap schedule, List<List<int>> taskClumps) {
        foreach (List<int> clump in taskClumps) {
            if (geneticRandom.Next(0, 100) < mutationChance) {
                int clumpSize = clump.Count;
                int possibleStart = FindExactFitSpot(schedule, clumpSize);

                if (possibleStart > -1)
                {
                    int newStartIndex = possibleStart;
                    for (int i = 0; i < clumpSize; i++)
                    {
                        schedule.mutateBit(clump[i]); 
                        schedule.mutateBit(newStartIndex + i); 
                    }
                }
            }
        }
        return schedule;
    }
    public ScheduleBitMap MutateCombineLoneSegments(ScheduleBitMap schedule, List<List<int>> taskClumps) {
        List<int> loneSegments = new List<int>();
        if (geneticRandom.Next(0, 100) < mutationChance) {
            foreach (List<int> clump in taskClumps) {
                if (clump.Count == 1) {
                    loneSegments.Add(clump[0]);
                }
            }
            if (loneSegments.Count == 1) {
                schedule = MutateSingleBit(schedule, loneSegments[0]);
            }
            else if (loneSegments.Count > 1) {
                MutateSingleClump(schedule, loneSegments.Take(geneticRandom.Next(0, loneSegments.Count)).ToList());
            }
        }
        return schedule;
    }

    // Pseudocode or a helper method:
    public int FindExactFitSpot(ScheduleBitMap schedule, int clumpSize)
    {
        // Get the total length of the schedule
        int scheduleSize = schedule.getSchedule().Length;

        // If the clump can't fit at all, return immediately
        if (clumpSize > scheduleSize) 
            return -1;

        // Generate one random start index (only up to scheduleSize - clumpSize)
        int randomStart = ScheduleMutator.geneticRandom.Next(0, scheduleSize - clumpSize + 1);

        // Check if all bits from randomStart..randomStart + clumpSize-1 are free
        for (int offset = 0; offset < clumpSize; offset++)
        {
            if (schedule.getBitValue(randomStart + offset))
            {
                // If any bit is occupied, fail immediately
                return -1;
            }
        }

        // If we got here, all bits in the clump are free
        return randomStart;
    }

    public List<List<int>> getTaskClumps(ScheduleBitMap schedule) {
        List<List<int>> taskClumps = new List<List<int>>();
        List<int> currentClump = new List<int>();
        List<int> sortedTaskIndexes = schedule.taskIndexes.ToList();
        sortedTaskIndexes.Sort();
        for (int i = 0; i < schedule.taskIndexes.Count; i++) {
            currentClump.Add(sortedTaskIndexes[i]);
            if (i >= schedule.taskIndexes.Count-1) {
                taskClumps.Add(currentClump);
            }
            else if (sortedTaskIndexes[i] + 1 != sortedTaskIndexes[i+1]) {
                taskClumps.Add(currentClump);
                currentClump = new List<int>();
            }
        }
        return taskClumps;
    }
}