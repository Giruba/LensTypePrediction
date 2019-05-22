using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.IO;

namespace LensTypePredictions
{
    class Program
    {
        public class LensDataModel
        {
            [Column("0")]
            public float SerialNo;

            [Column("1")]
            public float AgeTypeOfPatient;

            [Column("2")]
            public float SpectaclePrescribed_Myop_Hyper;

            [Column("3")]
            public float PresenceOfAstigmatism;

            [Column("4")]
            public float TearProductionRate;

            [Column("5")]
            [ColumnName("Label")]
            public float Label;
        }

        public class LensTypePrediction
        {
            [ColumnName("PredictedLabel")]
            public float PredictedLabels;
        }

        static void Main(string[] args)
        {

            var pipeline = new LearningPipeline();
            string dataPath = Path.Combine(Environment.CurrentDirectory, "lenses.data.txt");
            pipeline.Add(new TextLoader(dataPath).CreateFrom<LensDataModel>(separator: ','));
            pipeline.Add(new Dictionarizer("Label"));
            pipeline.Add(new ColumnConcatenator("Features", "AgeTypeOfPatient", "SpectaclePrescribed_Myop_Hyper", "PresenceOfAstigmatism", "TearProductionRate"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
            var model = pipeline.Train<LensDataModel, LensTypePrediction>();
            var prediction = model.Predict(new LensDataModel()
            {
                AgeTypeOfPatient = 1,
                SpectaclePrescribed_Myop_Hyper = 2,
                PresenceOfAstigmatism = 1,
                TearProductionRate = 2,
            });

            Console.WriteLine($"Predicted Lens for the patient is: {prediction.PredictedLabels}");
            Console.ReadKey();
        }
    }
}
