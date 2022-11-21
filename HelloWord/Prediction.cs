using Microsoft.ML.Data;

public class Prediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}