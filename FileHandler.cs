using System;
using System.IO;
using System.Text;
using MathNet.Numerics.Financial;
using MathNet.Numerics.LinearAlgebra;

public class BinaryFileHandler
{
    public Dictionary<int, string> intToBinaryMap = new Dictionary<int, string>
    {
        { -2, "000" },
        { -1, "001" },
        {  0, "010" },
        {  1, "011" },
        {  2, "100" }
    };
    public void SaveToBinaryFile(byte[] data, string relativeFilePath)
    {
        // Get the current directory (bin\Debug\net8.0)
        string currentDirectory = AppDomain.CurrentDomain.BaseDirectory;

        // Navigate up to the project root
        string projectDirectory = Path.GetFullPath(Path.Combine(currentDirectory, @"..\..\.."));

        // Combine with the relative file path
        string fullPath = Path.Combine(projectDirectory, relativeFilePath);

        // Ensure the directory exists
        string directory = Path.GetDirectoryName(fullPath);
        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        // Append data to the file, create if it doesn't exist
        using (FileStream fs = new FileStream(fullPath, FileMode.Append, FileAccess.Write))
        {
            fs.Write(data, 0, data.Length);
        }
    }

    public string SaveDataInfo(ScheduleBitMap schedule, GeneticAlgorithmGenerate geneticAlgorithm) {
        string scheduleBase = string.Concat(schedule.getBaseSchedule().Cast<bool>().Select(bit => bit ? "1" : "0"));
        string scheduleResult = ConvertSegmentDistributionToBinary(geneticAlgorithm);
        string taskSegments = new string('1', schedule.taskIndexes.Count).PadRight(12, '0');
        string preferredTimes = Convert2DArrayToBinary(geneticAlgorithm.fitnessCore.timeOfDayPreferences);
        string preferredDays = Convert1DArrayToBinary(geneticAlgorithm.fitnessCore.preferredDays);

        return scheduleBase + taskSegments + preferredTimes + preferredDays + scheduleResult;
    }

    public string ConvertSegmentDistributionToBinary(GeneticAlgorithmGenerate geneticAlgorithm) {
        string finalScheduleString = "";
        List<double> completeSegmentDistribution = new List<double>();
        for (int i = 0; i < geneticAlgorithm.scheduleSize; i++) {
            completeSegmentDistribution.Add(geneticAlgorithm.segmentScores.ContainsKey(i) ? geneticAlgorithm.segmentScores[i] : 0.00);
        }

        foreach (var segment in completeSegmentDistribution) {
            finalScheduleString += DoubleToIeee754Binary(segment).ToString();
        }
        return finalScheduleString;
    }

    public Tuple<List<NeuralScheduleHandler>, List<Matrix<double>>, List<Matrix<double>>> FileToDataSet(string relativeFilePath)
    {
        List<string> dataEntries = new List<string>();
        string currentDirectory = AppDomain.CurrentDomain.BaseDirectory;
        string projectDirectory = Path.GetFullPath(Path.Combine(currentDirectory, @"..\..\.."));
        string fullPath = Path.Combine(projectDirectory, relativeFilePath);

        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException($"The file {fullPath} does not exist.");
        }

        using (FileStream fs = new FileStream(fullPath, FileMode.Open, FileAccess.Read))
        using (BinaryReader br = new BinaryReader(fs))
        {
            int entrySize = 10935; //351
            long totalEntries = fs.Length / entrySize;

            if (fs.Length % entrySize != 0)
            {
                throw new Exception("File contains incomplete data entries.");
            }

            for (int i = 0; i < totalEntries; i++)
            {
                byte[] entryBytes = br.ReadBytes(entrySize);
                string binaryString = ByteArrayToBinaryString(entryBytes);
                dataEntries.Add(binaryString);
            }
        }

        List<NeuralScheduleHandler> dataExamples = new List<NeuralScheduleHandler>();
        List<Matrix<double>> xTrain = new List<Matrix<double>>();
        List<Matrix<double>> yTrain = new List<Matrix<double>>();

        foreach (string dataString in dataEntries) {
            NeuralScheduleHandler dataExample = new NeuralScheduleHandler(dataString);
            dataExample.CreateMatrixs();
            xTrain.Add(dataExample.inputData);
            yTrain.Add(dataExample.outputData);
            dataExamples.Add(dataExample);
        }

        return new Tuple<List<NeuralScheduleHandler>, List<Matrix<double>>, List<Matrix<double>>>(dataExamples, xTrain, yTrain);
    }
    public static string DoubleToIeee754Binary(double value) // 64-bit double
    {
        // Get the 8 bytes that represent the double in memory (little-endian on most machines)
        byte[] bytes = BitConverter.GetBytes(value);
        
        // Convert each byte to its 8-bit representation in reverse order (for big-endian display)
        // If you want the least significant bits at the end of the string, remove Reverse().
        var bits = bytes.Reverse()
                        .Select(b => Convert.ToString(b, 2).PadLeft(8, '0'));
        
        // Join them all into a single string
        return string.Join("", bits);
    }
    public static string ByteArrayToBinaryString(byte[] bytes)
    {
        StringBuilder sb = new StringBuilder();

        foreach (var b in bytes)
        {
            sb.Append(Convert.ToString(b, 2).PadLeft(8, '0'));
        }

        return sb.ToString();
    }
    public string Convert2DArrayToBinary(List<List<int>> array)
    {
        StringBuilder sb = new StringBuilder();

        foreach (var row in array)
        {
            foreach (var value in row)
            {
                if (intToBinaryMap.ContainsKey(value))
                {
                    sb.Append(intToBinaryMap[value]);
                }
                else
                {
                    throw new ArgumentException($"Value {value} is out of the allowed range (-2 to 2).");
                }
            }
        }

        return sb.ToString();
    }

    public string Convert1DArrayToBinary(List<int> array)
    {
        StringBuilder sb = new StringBuilder();

        foreach (var value in array)
        {
            if (intToBinaryMap.ContainsKey(value))
            {
                sb.Append(intToBinaryMap[value]);
            }
            else
            {
                throw new ArgumentException($"Value {value} is out of the allowed range (-2 to 2).");
            }
        }

        return sb.ToString();
    }

    public byte[] BinaryStringToByteArray(string binary)
    {
        // Calculate the number of bytes needed
        int numBytes = (binary.Length + 7) / 8;

        // Pad the binary string with '0's to make its length a multiple of 8
        binary = binary.PadRight(numBytes * 8, '0');

        byte[] bytes = new byte[numBytes];
        for (int i = 0; i < numBytes; i++)
        {
            string byteString = binary.Substring(8 * i, 8);
            bytes[i] = Convert.ToByte(byteString, 2);
        }

        return bytes;
    }
}
