using System;
using System.IO;

public class MNISTReader
{
    public static (byte[][] Images, byte[] Labels) LoadMNIST(string imagePath, string labelPath)
    {
        byte[] labels = ReadLabels(labelPath);
        byte[][] images = ReadImages(imagePath, labels.Length);
        return (images, labels);
    }

    private static byte[] ReadLabels(string labelPath)
    {
        using (var fs = new FileStream(labelPath, FileMode.Open, FileAccess.Read))
        using (var br = new BinaryReader(fs))
        {
            // Read magic number and number of labels
            int magic = ReadBigEndianInt32(br);
            int numLabels = ReadBigEndianInt32(br);
            byte[] labels = br.ReadBytes(numLabels);
            return labels;
        }
    }

    private static byte[][] ReadImages(string imagePath, int expectedCount)
    {
        using (var fs = new FileStream(imagePath, FileMode.Open, FileAccess.Read))
        using (var br = new BinaryReader(fs))
        {
            int magic = ReadBigEndianInt32(br);
            int numImages = ReadBigEndianInt32(br);
            int numRows = ReadBigEndianInt32(br);
            int numCols = ReadBigEndianInt32(br);

            if (numImages != expectedCount)
            {
                throw new Exception("Mismatch between label and image count.");
            }

            byte[][] images = new byte[numImages][];

            for (int i = 0; i < numImages; i++)
            {
                byte[] image = br.ReadBytes(numRows * numCols);
                images[i] = image;
            }

            return images;
        }
    }

    private static int ReadBigEndianInt32(BinaryReader br)
    {
        byte[] bytes = br.ReadBytes(4);
        if (BitConverter.IsLittleEndian)
            Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }
}

// Example usage:
// var (trainImages, trainLabels) = MNISTReader.LoadMNIST("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
// var (testImages, testLabels) = MNISTReader.LoadMNIST("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");
