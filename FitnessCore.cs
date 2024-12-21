using System.Security.Cryptography.X509Certificates;
using System.Security.Principal;
using MathNet.Numerics.LinearAlgebra;

public class FitnessCore {
    public FitnessCore() {
    }
    public double clumpingMultiplyer = 30;
    public double taskSplitMultiplier = 3000;
    public double daySplitMultiplier = 5000;
    public double minTaskScorePenalty = 1000;
    public int minTaskSize = 2;
    public List<List<double>> TODMultiplyers = new List<List<double>> { //These numbers have to be extreme i.e ~50 change by 20
        new List<double> { 50, 50, 50, 50 }, //0
        new List<double> { 50, 50, 50, 50 }, //1
        new List<double> { 50, 50, 50, 50 }, //2
        new List<double> { 50, 50, 50, 50 }, //3
        new List<double> { 50, 50, 50, 50 }, //4
        new List<double> { 50, 50, 50, 50 }, //5
        new List<double> { 50, 50, 50, 50 }, //6
    };

    /*
    Clump Score
    Tries to get 3 hours of work in a row
    The slope as lim x=12 (3 hours) from the right is greater than that of the slope as x=12 from the left
        I.E 2 clumps of 2 hours of work is better than 1 clump of 4 hours of work 
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
                    clumpSize-=2;
            }
            clumpScore += CalculateClumpScore(clumpSize);
        }
        return clumpScore;
    }
    public double CalculateClumpScore(int x) {
        double y = 0;

        if (x <= 12)
        {
            y = 1 / (1.0 + Math.Pow((x-12) / 12.0, 2)) * clumpingMultiplyer;
        }
        else
        {
            y = 1 / (1.0 + Math.Pow((x-12) / 8.0, 2))  * clumpingMultiplyer;
        }
        return y;
    }
    public double RunTODScore(ScheduleBitMap schedule) {
        double todScore = 0;
        int currentTODSection;
        int currentDay;
        for (int i = 0; i < schedule.getSchedule().Length; i++) {
            currentTODSection = calculateCurrentTOD(i) % 4;
            currentDay = (int)(calculateCurrentTOD(i) / 4);
            if (schedule.getBitValue(i)) {
                todScore += 1 * TODMultiplyers[currentDay][currentTODSection];
            }
        }
        return todScore;
    }
    public int calculateCurrentTOD(int i) {
        return (int)(i / 24);
    }
    public double RunSplitTaskScore(ScheduleBitMap schedule) {
        List<int> sortedTaskIndexs = schedule.taskIndexs.ToList();
        sortedTaskIndexs.Sort();
        int splitDifference = sortedTaskIndexs[sortedTaskIndexs.Count-1] - sortedTaskIndexs[0] + 1; //+1 to account for 0 index
        return CalculateSplitScore(splitDifference, taskSplitMultiplier);
    }
    public double CalculateSplitScore(int x, double multiplier) {
        return 1 / (1.0 + Math.Pow((x-1) / 2.5, 4)) * multiplier;
    }
    public double RunDaySpread(ScheduleBitMap schedule) {
        int dayCount = 1;
        List<int> sortedTaskIndexs = schedule.taskIndexs.ToList();
        sortedTaskIndexs.Sort();
        for (int i = 0; i < sortedTaskIndexs.Count-1; i++) {
            if ((int)(calculateCurrentTOD(sortedTaskIndexs[i]) / 4) != (int)(calculateCurrentTOD(sortedTaskIndexs[i+1]) / 4)) {
                dayCount++;
            }
        }
        return CalculateSplitScore(dayCount, daySplitMultiplier);
    }
    public double RunMinimumTaskSize(ScheduleBitMap schedule) {
        int currentTaskSize = 1;
        double minTaskScore = 0;
        List<int> sortedTaskIndexs = schedule.taskIndexs.ToList();
        sortedTaskIndexs.Sort();
        for (int i = 0; i < sortedTaskIndexs.Count()-1; i++) {
            if (sortedTaskIndexs[i] + 1 != sortedTaskIndexs[i+1]) {
                minTaskScore -= currentTaskSize < minTaskSize ? minTaskScorePenalty : 0;
                currentTaskSize = 1;
            }
            else {
                currentTaskSize++;
            }
        }
        return minTaskScore;
    }
    public double FitnessFunction(ScheduleBitMap schedule) {
        // schedule.fitness += RunClumpScore(schedule);
        // schedule.fitness += RunTODScore(schedule);
        schedule.fitness += RunSplitTaskScore(schedule);
        // schedule.fitness += RunDaySpread(schedule);
        schedule.fitness += RunMinimumTaskSize(schedule);
        return schedule.fitness;
    }
}