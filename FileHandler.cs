using System;
using System.IO;
using System.Text;

public class BinaryFileHandler
{
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

    public byte[] ReadFromBinaryFile(string relativeFilePath)
    {
        // Get the current directory (bin\Debug\net8.0)
        string currentDirectory = AppDomain.CurrentDomain.BaseDirectory;

        // Navigate up to the project root
        string projectDirectory = Path.GetFullPath(Path.Combine(currentDirectory, @"..\..\.."));

        // Combine with the relative file path
        string fullPath = Path.Combine(projectDirectory, relativeFilePath);

        // Check if file exists
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException($"The file {fullPath} does not exist.");
        }

        // Read all bytes from the file
        return File.ReadAllBytes(fullPath);
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



    public List<NeuralScheduleHandler> FileToDataSet(string relativeFilePath)
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
            int entrySize = 351;
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

        foreach (string dataString in dataEntries) {
            dataExamples.Add(new NeuralScheduleHandler(dataString));
        }

        return dataExamples;
    }


}
