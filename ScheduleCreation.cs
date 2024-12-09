using System.Reflection;
using MathNet.Numerics.LinearAlgebra;

public class ScheduleCreation {
    private Random scheduleCreationRandom = new Random();
    public int days {get; set;}
    public int segments {get; private set;}
    public ScheduleCreation(int days) {
        this.days = days;
        this.segments = days * 96;

    }

    public Matrix<double> generateSchedule(int scheduleItems) {
        Matrix<double> scheduleData = Matrix<double>.Build.Dense(scheduleItems, this.segments, (r, c) => scheduleCreationRandom.Next(0, 2));
        return scheduleData;
    }
}