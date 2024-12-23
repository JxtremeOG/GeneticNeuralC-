using System.Security.Cryptography.X509Certificates;
using System.Security.Principal;
using MathNet.Numerics.LinearAlgebra;

public class FitnessCore {
    public FitnessCore() {
    }
    public HashSet<double> clumpScores = new HashSet<double>();
    public double clumpingMultiplier = 20;
    public double taskSplitMultiplier = 6000;
    public double daySplitMultiplier = 13000;
    public double taskClumpScoreMultiplier = 800;
    public int minTaskSize = 2;
    public List<List<double>> TODMultipliers = new List<List<double>> { //change by 25
        new List<double> { 25, 25, -75, 25 }, //0
        new List<double> { 25, 25, -75, 25 }, //1
        new List<double> { 25, 25, -75, 25 }, //2
        new List<double> { 25, 25, -75, 25 }, //3
        new List<double> { 25, 25, -75, 25 }, //4
        new List<double> { 25, 25, -75, 25 }, //5
        new List<double> { 25, 25, -75, 25 }, //6
    };

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
        for (int i = 0; i < schedule.getSchedule().Length; i++) {
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
    public double RunTODScore(ScheduleBitMap schedule) {
        double todScore = 0;
        int currentTODSection;
        int currentDay;
        for (int i = 0; i < schedule.getSchedule().Length; i++) {
            currentTODSection = calculateCurrentTOD(i) % 4;
            currentDay = (int)(calculateCurrentTOD(i) / 4);
            if (schedule.getBitValue(i)) {
                todScore += 1 * TODMultipliers[currentDay][currentTODSection];
            }
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
        }
        return taskClumpScore;
    }
    public double FitnessFunction(ScheduleBitMap schedule) {
        schedule.fitness += RunClumpScore(schedule);
        schedule.fitness += RunTODScore(schedule);
        schedule.fitness += RunSplitTaskScore(schedule);
        schedule.fitness += RunDaySpread(schedule);
        schedule.fitness += RunTaskClump(schedule);
        return schedule.fitness;
    }
}