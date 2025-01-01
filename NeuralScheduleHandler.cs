using System.Collections;
using MathNet.Numerics.LinearAlgebra;

public class NeuralScheduleHandler {
    public Dictionary<string, int> binaryToIntMap = new Dictionary<string, int>
    {
        { "000", -2 },
        { "001", -1 },
        { "010",  0 },
        { "011",  1 },
        { "100",  2 }
    };
    public BitArray baseSchedule;
    public BitArray taskSegments;
    public string finalScheduleString;
    public List<double> timeOfDayPreferences;
    public List<double> dayPreferences;

    public Matrix<double> inputData;
    public Matrix<double> outputData;
    public NeuralScheduleHandler(string binaryString) {
        //1344 Base Schedule
        //12 task segments
        //84 time of days
        //21 days
        //86016 final schedule (high because decimal to binary)
        // == 87477
        if (binaryString.Length != 87480)
            throw new Exception($"Unexpected data example length");

        string baseScheduleString = binaryString.Substring(0, 1344);
        string taskSegmentsString = binaryString.Substring(1344, 12);
        string timeOfDayString = binaryString.Substring(1344+12, 84);
        string daysString = binaryString.Substring(1344+12+84, 21);
        finalScheduleString = binaryString.Substring(1344+12+84+21, 86016);

        baseSchedule = BinaryStringToBitArray(baseScheduleString);
        taskSegments = BinaryStringToBitArray(taskSegmentsString);

        timeOfDayPreferences = MapBinaryStringToList(timeOfDayString);
        dayPreferences = MapBinaryStringToList(daysString);
    }

    public void CreateMatrixs() {
        List<double> baseShceduleList = baseSchedule.Cast<bool>().Select(bit => bit ? 0.0 : 1.0).ToList(); //Also reverse so 0 == taken spot
        List<double> taskSegmentsList = taskSegments.Cast<bool>().Select(bit => bit ? 1.0 : 0.0).ToList();
        List<double> masterCombined = baseShceduleList
            .Concat(taskSegmentsList)
            .Concat(timeOfDayPreferences)
            .Concat(dayPreferences)
            .ToList();

        inputData = Matrix<double>.Build.Dense(
            1,                           // number of rows
            masterCombined.Count,           // number of columns
            (r, c) => masterCombined[c]     // fill function
        );

        List<double> finalScheduleList = Ieee754BinaryToDoubles(finalScheduleString);

        outputData = Matrix<double>.Build.Dense(
            1,                           // number of rows
            finalScheduleList.Count,           // number of columns
            (r, c) => finalScheduleList[c]     // fill function
        );
    }

    public static List<double> Ieee754BinaryToDoubles(string bitString)
    {
        // The total length must be a multiple of 64
        if (bitString.Length % 64 != 0)
            throw new ArgumentException(
                "The bit string length must be a multiple of 64.", nameof(bitString));

        int count = bitString.Length / 64;
        var doubles = new List<double>(count);

        for (int i = 0; i < count; i++)
        {
            // Extract the 64-bit segment
            string chunk = bitString.Substring(i * 64, 64);

            // Convert that 64-bit chunk to a double
            double d = Ieee754BinaryToDouble(chunk);
            doubles.Add(d);
        }

        return doubles;
    }

    /// <summary>
    /// Converts a 64-bit binary string (as output by DoubleToIeee754Binary)
    /// into a double. The string should be exactly 64 characters ('0' or '1').
    /// </summary>
    public static double Ieee754BinaryToDouble(string bitString)
    {
        // Expect exactly 64 characters for a 64-bit double
        if (bitString.Length != 64)
            throw new ArgumentException("Binary string must be exactly 64 bits.", nameof(bitString));

        // Break the 64-bit string into 8 groups of 8 bits each
        var byteList = new List<byte>(8);
        for (int i = 0; i < 64; i += 8)
        {
            // Extract 8 bits
            string byteChunk = bitString.Substring(i, 8);
            byte b = Convert.ToByte(byteChunk, 2);
            byteList.Add(b);
        }

        // Remember: DoubleToIeee754Binary reversed the bytes for display, 
        // so we reverse them back here to restore the original little-endian order
        byteList.Reverse();

        // Convert the bytes to a double
        return BitConverter.ToDouble(byteList.ToArray(), 0);
    }

    public BitArray BinaryStringToBitArray(string binaryString)
    {
        if (string.IsNullOrEmpty(binaryString))
            throw new ArgumentException("Binary string cannot be null or empty.");

        // Convert each character to a boolean: '1' => true, '0' => false
        bool[] boolArray = binaryString.Select(c =>
        {
            if (c == '1') return true;
            if (c == '0') return false;
            throw new ArgumentException($"Invalid character '{c}' in binary string. Only '0' and '1' are allowed.");
        }).ToArray();

        return new BitArray(boolArray);
    }
    public List<double> MapBinaryStringToList(string binaryString)
    {
        const int segmentLength = 3;
        
        if (binaryString.Length % segmentLength != 0)
        {
            throw new ArgumentException($"Binary string length must be a multiple of {segmentLength}.");
        }

        List<double> preferences = new List<double>();
        
        for (int i = 0; i < binaryString.Length; i += segmentLength)
        {
            string segment = binaryString.Substring(i, segmentLength);
            
            if (!binaryToIntMap.TryGetValue(segment, out int mappedValue))
            {
                throw new KeyNotFoundException($"The binary segment '{segment}' is not defined in the binaryToIntMap.");
            }
            
            preferences.Add(mappedValue);
        }
        
        return preferences;
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