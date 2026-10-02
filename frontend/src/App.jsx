
import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setResult(null);
    setError("");
  };

  const handlePredict = async () => {
    if (!file) {
      setError("Please select an MRI file first.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");

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
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);

    } catch (err) {
      console.error("Prediction error:", err);
      setError(
        "Unable to process the MRI. Please check your connection and try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const downloadImage = () => {
    if (!result) return;

    const link = document.createElement("a");
    link.href = `data:image/png;base64,${result.image}`;
    link.download = "brain_tumor_prediction.png";
    link.click();
  };

  const generateReport = () => {
    if (!result) return "";

    return `The uploaded FLAIR MRI volume, named ${result.filename}, was successfully processed using the U-Net segmentation model with a ResNet34 encoder.

The MRI volume contained ${result.total_slices} slices, with dimensions of ${result.volume_shape[0]} × ${result.volume_shape[1]} × ${result.volume_shape[2]} voxels.

The model generated predicted segmentation regions across ${result.predicted_tumor_slices} slices. Among these, slice ${result.largest_tumor_slice} contained the largest predicted segmentation region, covering ${result.largest_slice_area_pixels.toLocaleString()} pixels.

Across the complete volume, the model predicted ${result.total_predicted_tumor_voxels.toLocaleString()} voxels as part of the segmentation mask.

These measurements describe the model's predicted output only. They do not establish a clinical diagnosis, determine tumor type, or indicate malignancy.`;
  };

  return (
    <div className="app-layout">

      {/* LEFT SIDEBAR */}
      <aside className="sidebar">

        <div className="brand">
          <div className="brand-icon">✚</div>
          <div>
            <h2>NeuroScan</h2>
            <span>Brain Imaging Analysis</span>
          </div>
        </div>

        <div className="sidebar-section">
          <p className="section-label">MRI ANALYSIS</p>

          <label className="upload-area">
            <div className="upload-icon">↑</div>
            <strong>Upload FLAIR MRI</strong>
            <span>Select a NIfTI volume</span>
            <span className="file-formats">.nii / .nii.gz</span>

            <input
              type="file"
              accept=".nii,.nii.gz"
              onChange={handleFileChange}
            />
          </label>

          {file && (
            <div className="selected-file">
              <span>Selected file</span>
              <strong>{file.name}</strong>
            </div>
          )}

          <button
            className="predict-button"
            onClick={handlePredict}
            disabled={!file || loading}
          >
            {loading ? "Processing MRI..." : "Run Segmentation"}
          </button>

          {loading && (
            <p className="loading-text">
              Processing MRI slices. This may take a few minutes.
            </p>
          )}

          {error && <p className="error-message">{error}</p>}
        </div>

        <div className="sidebar-section model-info">
          <p className="section-label">MODEL INFORMATION</p>

          <div className="info-item">
            <span>Architecture</span>
            <strong>U-Net</strong>
          </div>

          <div className="info-item">
            <span>Encoder</span>
            <strong>ResNet34</strong>
          </div>

          <div className="info-item">
            <span>Input modality</span>
            <strong>FLAIR</strong>
          </div>

          <div className="info-item">
            <span>Segmentation</span>
            <strong>Whole Tumor</strong>
          </div>
        </div>

        <div className="sidebar-footer">
          Academic Research Project<br />
          Not for clinical diagnosis
        </div>

      </aside>

      {/* MAIN CONTENT */}
      <main className="main-content">

        <header className="topbar">
          <div>
            <h1>Brain Tumor Segmentation</h1>
            <p>AI-assisted MRI segmentation and analysis</p>
          </div>

          <span className="status-badge">
            Research Demo
          </span>
        </header>

        {!result && !loading && (
          <section className="welcome-card">
            <div className="welcome-icon">🧠</div>
            <h2>Welcome to NeuroScan</h2>
            <p>
              Upload a FLAIR MRI volume to generate a predicted
              brain tumor segmentation mask and its analysis report.
            </p>
            <span>Select your MRI file from the left panel to begin.</span>
          </section>
        )}

        {loading && (
          <section className="welcome-card">
            <div className="loader"></div>
            <h2>Analyzing MRI Volume</h2>
            <p>
              The segmentation model is processing the MRI slices.
              Please wait.
            </p>
          </section>
        )}

        {result && (
          <div className="results-container">

            <section className="result-card">
              <div className="card-heading">
                <div>
                  <h2>Segmentation Visualization</h2>
                  <p>Predicted region highlighted in red</p>
                </div>
                <span className="result-tag">Completed</span>
              </div>

              <div className="image-container">
                <img
                  src={`data:image/png;base64,${result.image}`}
                  alt="Predicted brain tumor segmentation overlay"
                />
              </div>

              <div className="image-caption">
                <span className="red-dot"></span>
                Red: Predicted segmentation region
              </div>

              <button
                className="download-button"
                onClick={downloadImage}
              >
                ↓ Download Segmentation Image
              </button>
            </section>

            <section className="report-card">

              <div className="card-heading">
                <div>
                  <h2>Prediction Report</h2>
                  <p>Summary of the model's segmentation output</p>
                </div>
                <span className="report-icon">▤</span>
              </div>

              <div className="statistics-grid">

                <div className="stat-box">
                  <span>Total Slices</span>
                  <strong>{result.total_slices}</strong>
                </div>

                <div className="stat-box">
                  <span>Predicted Slices</span>
                  <strong>{result.predicted_tumor_slices}</strong>
                </div>

                <div className="stat-box">
                  <span>Largest Region</span>
                  <strong>Slice {result.largest_tumor_slice}</strong>
                </div>

                <div className="stat-box">
                  <span>Predicted Voxels</span>
                  <strong>
                    {result.total_predicted_tumor_voxels.toLocaleString()}
                  </strong>
                </div>

              </div>

              <div className="written-report">
                <h3>Analysis Summary</h3>
                <p>{generateReport()}</p>
              </div>

              <div className="report-details">
                <div>
                  <span>Input file</span>
                  <strong>{result.filename}</strong>
                </div>

                <div>
                  <span>Volume dimensions</span>
                  <strong>
                    {result.volume_shape.join(" × ")}
                  </strong>
                </div>

                <div>
                  <span>Largest slice area</span>
                  <strong>
                    {result.largest_slice_area_pixels.toLocaleString()} pixels
                  </strong>
                </div>
              </div>

            </section>

            <div className="disclaimer">
              <strong>Research Disclaimer</strong>
              <p>
                This segmentation is generated by an academic deep learning
                model. The predicted regions are not a medical diagnosis
                and should not be used for clinical decision-making.
                Results require interpretation by qualified medical
                professionals.
              </p>
            </div>

          </div>
        )}

      </main>
    </div>
  );
}

export default App;
