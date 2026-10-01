
import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setResult(null);
  };

  const handlePredict = async () => {
    if (!file) {
      alert("Please select an MRI file first.");
      return;
    }

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(
          `Server returned ${response.status}`
        );
      }

      const data = await response.json();

      setResult(data);

    } catch (error) {
      console.error("Prediction error:", error);
      alert("Something went wrong while processing the MRI.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="container">

        <h1>Brain Tumor Segmentation</h1>

        <p className="description">
          Upload a FLAIR MRI scan to visualize the
          predicted tumor region.
        </p>

        <div className="upload-box">

          <input
            type="file"
            accept=".nii,.nii.gz"
            onChange={handleFileChange}
          />

          {file && (
            <p className="file-name">
              Selected file: {file.name}
            </p>
          )}

          <button
            onClick={handlePredict}
            disabled={!file || loading}
          >
            {loading
              ? "Processing MRI..."
              : "Predict Tumor"}
          </button>

        </div>

        {result && (
          <div className="result">

            <h2>Prediction Result</h2>

            <img
              src={`data:image/png;base64,${result.image}`}
              alt="Brain tumor prediction"
              className="result-image"
            />

            <div className="report">

              <h2>Prediction Report</h2>

              <div className="report-row">
                <span>Input File</span>
                <strong>{result.filename}</strong>
              </div>

              <div className="report-row">
                <span>MRI Dimensions</span>
                <strong>
                  {result.volume_shape[0]} ×{" "}
                  {result.volume_shape[1]} ×{" "}
                  {result.volume_shape[2]}
                </strong>
              </div>

              <div className="report-row">
                <span>Total Slices Processed</span>
                <strong>
                  {result.total_slices}
                </strong>
              </div>

              <div className="report-row">
                <span>Slices With Predicted Tumor</span>
                <strong>
                  {result.predicted_tumor_slices}
                </strong>
              </div>

              <div className="report-row">
                <span>Largest Predicted Region</span>
                <strong>
                  Slice {result.largest_tumor_slice}
                </strong>
              </div>

              <div className="report-row">
                <span>Tumor Area on Selected Slice</span>
                <strong>
                  {result.largest_slice_area_pixels.toLocaleString()} pixels
                </strong>
              </div>

              <div className="report-row">
                <span>Total Predicted Tumor Voxels</span>
                <strong>
                  {result.total_predicted_tumor_voxels.toLocaleString()}
                </strong>
              </div>

            </div>

            <a
              href={`data:image/png;base64,${result.image}`}
              download="brain_tumor_prediction.png"
              className="download-button"
            >
              Download Result
            </a>

          </div>
        )}

        <p className="disclaimer">
          This tool is an academic demonstration and is
          not intended for medical diagnosis.
        </p>

      </div>
    </div>
  );
}

export default App;