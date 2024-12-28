using System.Security.Cryptography.X509Certificates;
using System.Security.Principal;
using MathNet.Numerics.LinearAlgebra;

public class FitnessCore {
    public FitnessCore() {
    }
    public HashSet<double> clumpScores = new HashSet<double>();
    public double clumpingMultiplier = 10;
    public double taskSplitMultiplier = 3000;
    public double daySplitMultiplier = 6500;
    public double taskClumpScoreMultiplier = 400;
    public double loneTaskScorePenalty = 200;
    public double switchTaskScorePenalty = 500;
    public int earlyTaskScoreMax = 1000;
    public int preferredDayScoreMax = 1000;
    public int minTaskSize = 2;
    public int timeOfDayScoreMax = 1000;
    public List<List<double>> timeOfDayPreferences = new List<List<double>> { // 1 is preferred 0 is neural and -2 is not preferred
        new List<double> { 0, 0, 0, 0 }, //0
        new List<double> { 0, 0, 0, 0 }, //1
        new List<double> { 0, 0, 0, 0 }, //2
        new List<double> { 0, 0, 0, 0 }, //3
        new List<double> { 0, 0, 0, 0 }, //4
        new List<double> { 0, 0, 0, 0 }, //5
        new List<double> { 0, 0, 0, 0 }, //6
    };

    public List<double> preferredDays = new List<double> { 0, 0, 0, 0, 0, 0, 0 };

    public double PopulationStdDev(IEnumerable<int> values)
    {
        double mean = values.Average();

        double variance = values
            .Select(v => Math.Pow(v - mean, 2))
            .Average();

        return Math.Sqrt(variance);
    }

    public void PreCalculateClumpScores(int scheduleLength) {
        for (int i = 0; i < scheduleLength; i++) {
            clumpScores.Add(CalculateClumpScore(i, clumpingMultiplier));
        }
    }

    public double CalculateClumpScore(int x, double multiplier) {
        double y = 0;

        if (x <= 12)
        {
            y = 1 / (1.0 + Math.Pow((x-12) / 12.0, 2)) * multiplier;
        }
        else
        {
            y = 1 / (1.0 + Math.Pow((x-12) / 8.0, 2))  * multiplier;
        }
        return y;
    }
    public int calculateCurrentTOD(int i) {
        return (int)(i / 24);
    }
    public double CalculateSplitScore(int x, double multiplier) {
        return 1 / (1.0 + Math.Pow((x-1) / 2.5, 4)) * multiplier;
    }

    /*
        Encourages working time to be clumped to 3 hour intervals ( 12 segments )
    */
    public double RunClumpScore(ScheduleBitMap schedule) {
        double clumpScore = 0;
        int clumpSize = 0;
        for (int i = 0; i < schedule.scheduleSize; i++) {
            if (schedule.getBitValue(i)) {
                if (clumpSize < 0) {
                    clumpSize = 0;
                }
                clumpSize++;
            }
            else {
                if (clumpSize > 24)
                    clumpSize = 0;
                else
                    clumpSize = clumpSize > 2 ? clumpSize-2 : 0;
            }
            // clumpScore += CalculateClumpScore(clumpSize);
            clumpScore += clumpScores.ElementAt(clumpSize);
        }
        return clumpScore;
    }
    /*
        Encourages tasks to be placed at preferred Times of day
    */
    public double RunTimeOfDayScore(ScheduleBitMap schedule) {
        double scorePerSegment = timeOfDayScoreMax / schedule.taskIndexes.Count;
        double todScore = 0;
        int currentTODSection;
        int currentDay;
        List<int> taskIndexes = schedule.taskIndexes.ToList();
        foreach (int taskIndex in taskIndexes) {
            currentTODSection = calculateCurrentTOD(taskIndex) % 4;
            currentDay = (int)(calculateCurrentTOD(taskIndex) / 4);
            todScore += timeOfDayPreferences[currentDay][currentTODSection] * scorePerSegment;
        }
        return todScore;
    }
    /*
        Encourages tasks to not be over split up
    */
    public double RunSplitTaskScore(ScheduleBitMap schedule) {
        List<int> sortedTaskIndexes = schedule.taskIndexes.ToList();
        sortedTaskIndexes.Sort();
        int splitDifference = sortedTaskIndexes[sortedTaskIndexes.Count-1] - sortedTaskIndexes[0] + 1; //+1 to account for 0 index
        return CalculateSplitScore(splitDifference, taskSplitMultiplier);
    }
    /*
        Encourages tasks of the same type to not spread to far over multiple days
    */
    public double RunDaySpread(ScheduleBitMap schedule) {
        int dayCount = 1;
        List<int> sortedTaskIndexs = schedule.taskIndexes.ToList();
        sortedTaskIndexs.Sort();
        for (int i = 0; i < sortedTaskIndexs.Count-1; i++) {
            if ((int)(calculateCurrentTOD(sortedTaskIndexs[i]) / 4) != (int)(calculateCurrentTOD(sortedTaskIndexs[i+1]) / 4)) {
                dayCount++;
            }
        }
        return CalculateSplitScore(dayCount, daySplitMultiplier);
    }
    /*
        Encourages tasks of the same type to be clumped together up to 3 hours ( 12 segments )
    */
    public double RunTaskClump(ScheduleBitMap schedule) {
        int currentTaskSize = 1;
        double taskClumpScore = 0;
        List<int> sortedTaskIndexes = schedule.taskIndexes.ToList();
        sortedTaskIndexes.Sort();
        for (int i = 0; i < sortedTaskIndexes.Count()-1; i++) {
            if (sortedTaskIndexes[i] + 1 != sortedTaskIndexes[i+1]) {
                currentTaskSize = 1;
            }
            else {
                currentTaskSize++;
            }
            taskClumpScore += CalculateClumpScore(currentTaskSize, taskClumpScoreMultiplier);
            if (currentTaskSize < minTaskSize) {
                taskClumpScore -= loneTaskScorePenalty;
            }
        }
        return taskClumpScore;
    }
    /*
        Discourages switching from task to a previous event right back to task
    */
    public double RunSwitchScore(ScheduleBitMap schedule) {
        double switchScore = 0;
        List<int> sortedTaskIndexes = schedule.taskIndexes.ToList();
        sortedTaskIndexes.Sort();
        for (int i = 0; i < schedule.scheduleSize-2; i++) {
            if (schedule.getBitValue(i) && schedule.getBitValue(i+2) && sortedTaskIndexes.Contains(i+1) && !sortedTaskIndexes.Contains(i+2) && !sortedTaskIndexes.Contains(i)) {
                switchScore -= switchTaskScorePenalty;
            }
            if (sortedTaskIndexes.Contains(i+2) && sortedTaskIndexes.Contains(i) && !sortedTaskIndexes.Contains(i+1) && schedule.getBitValue(i+1)) {
                switchScore -= switchTaskScorePenalty;
            }
        }
        return switchScore;
    }
    /*
        Encourages tasks to be placed earlier in the schedule
    */
    public double RunEarlyScore(ScheduleBitMap schedule) {
        double scorePerSegment = earlyTaskScoreMax / schedule.taskIndexes.Count;
        double earlyScore = 0;
        foreach (int index in schedule.taskIndexes) {
            earlyScore += (schedule.scheduleSize - index) / (double)schedule.scheduleSize * scorePerSegment;
        }
        return earlyScore;
    }
    /*
        Encourages tasks to be placed on preferred days
    */
    public double RunPreferredDayScore(ScheduleBitMap schedule) {
        int currentDay;
        double scorePerSegment = preferredDayScoreMax / schedule.taskIndexes.Count;
        double preferredDayScore = 0;
        foreach (int index in schedule.taskIndexes) {
            currentDay = (int)(calculateCurrentTOD(index) / 4);
            preferredDayScore += preferredDays[currentDay] * scorePerSegment;
        }
        return preferredDayScore;
    }
    /*
        Encourages days to have an even spread of tasks
    */
    public double RunEvenDays(ScheduleBitMap schedule) {
        List<int> days = Enumerable.Repeat(0, schedule.scheduleSize / 96).ToList();
        int currentDay;
        for (int i = 0; i < schedule.scheduleSize; i++) {
            currentDay = (int)(calculateCurrentTOD(i) / 4);
            if (schedule.getBitValue(i))
                days[currentDay]++;
        }
        double standardDev = PopulationStdDev(days);
        schedule.scheduleDeviation = standardDev;
        return (48 - standardDev) * 21; //Roughly 1000 points max
    }
    
    public double FitnessFunction(ScheduleBitMap schedule) {
        schedule.fitness += RunClumpScore(schedule);
        schedule.fitness += RunTimeOfDayScore(schedule);
        schedule.fitness += RunSplitTaskScore(schedule);
        schedule.fitness += RunDaySpread(schedule);
        schedule.fitness += RunTaskClump(schedule);
        schedule.fitness += schedule.taskIndexes.Count > 1 ? RunSwitchScore(schedule) : 0; //only run if task size is greater than 1
        schedule.fitness += RunEarlyScore(schedule);
        schedule.fitness += RunPreferredDayScore(schedule);
        schedule.fitness += RunEvenDays(schedule);
        return schedule.fitness;
    }
}