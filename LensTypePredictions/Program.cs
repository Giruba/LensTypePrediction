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

        // IrisPrediction is the result returned from prediction operations
        public class LensTypePrediction
        {
            [ColumnName("PredictedLabel")]
            public float PredictedLabels;
        }

        static void Main(string[] args)
        {
            // STEP 2: Create a pipeline and load your data
            var pipeline = new LearningPipeline();

            // If working in Visual Studio, make sure the 'Copy to Output Directory' 
            // property of iris-data.txt is set to 'Copy always'
            string dataPath = Path.Combine(Environment.CurrentDirectory, "lenses.data.txt");
            pipeline.Add(new TextLoader(dataPath).CreateFrom<LensDataModel>(separator: ','));

            // STEP 3: Transform your data
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training
            pipeline.Add(new Dictionarizer("Label"));

            // Puts all features into a vector
            pipeline.Add(new ColumnConcatenator("Features", "AgeTypeOfPatient", "SpectaclePrescribed_Myop_Hyper", "PresenceOfAstigmatism", "TearProductionRate"));

            // STEP 4: Add learner
            // Add a learning algorithm to the pipeline. 
            // This is a classification scenario (What type of iris is this?)
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // Convert the Label back into original text (after converting to number in step 3)
            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // STEP 5: Train your model based on the data set
            var model = pipeline.Train<LensDataModel, LensTypePrediction>();

            // STEP 6: Use your model to make a prediction
            // You can change these numbers to test different predictions
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