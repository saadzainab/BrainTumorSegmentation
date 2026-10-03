import { useState } from "react";
import "./App.css";


function App() {

  // =========================================================
  // State
  // =========================================================

  const [file, setFile] = useState(null);

  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  // Current visualization:
  // original | mask | overlay
  const [activeView, setActiveView] = useState("overlay");

  // Current MRI slice displayed in the viewer
  const [currentSlice, setCurrentSlice] = useState(0);


  // =========================================================
  // File Selection
  // =========================================================

  const handleFileChange = (event) => {

    const selectedFile =
      event.target.files[0];

    setFile(
      selectedFile
    );

    setResult(
      null
    );

    setError(
      ""
    );

    setActiveView(
      "overlay"
    );

    setCurrentSlice(
      0
    );
  };


  // =========================================================
  // Run Prediction
  // =========================================================

  const handlePredict = async () => {

    if (!file) {

      setError(
        "Please select an MRI file first."
      );

      return;
    }


    setLoading(
      true
    );

    setResult(
      null
    );

    setError(
      ""
    );

    setActiveView(
      "overlay"
    );


    const formData =
      new FormData();

    formData.append(
      "file",
      file
    );


    try {

      const response =
        await fetch(
          "http://127.0.0.1:8000/predict",
          {
            method: "POST",
            body: formData,
          }
        );


      if (!response.ok) {

        let message =
          `Server error: ${response.status}`;

        try {

          const errorData =
            await response.json();

          if (errorData.detail) {
            message =
              errorData.detail;
          }

        } catch {
          // Keep default message
        }

        throw new Error(
          message
        );
      }


      const data =
        await response.json();


      console.log(
        "Prediction result:",
        data
      );


      setResult(
        data
      );


      // Start viewer at largest predicted region
      setCurrentSlice(
        Number(
          data.largest_tumor_slice
        )
      );


    } catch (err) {

      console.error(
        "Prediction error:",
        err
      );


      setError(
        err.message ||
        "Unable to process the MRI. Please check the backend server and try again."
      );


    } finally {

      setLoading(
        false
      );
    }
  };


  // =========================================================
  // Current Slice Object
  // =========================================================

  const getCurrentSliceData = () => {

    if (!result) {
      return null;
    }


    if (
      !result.slice_visualizations ||
      result.slice_visualizations.length === 0
    ) {

      return null;
    }


    return (
      result.slice_visualizations[
        currentSlice
      ] || null
    );
  };


  // =========================================================
  // Get Active Image
  // =========================================================

  const getActiveImage = () => {

    if (!result) {
      return "";
    }


    const sliceData =
      getCurrentSliceData();


    // -------------------------------------------------------
    // New interactive slice viewer
    // -------------------------------------------------------

    if (sliceData) {

      if (
        activeView === "original"
      ) {

        return (
          sliceData.original_image
        );
      }


      if (
        activeView === "mask"
      ) {

        return (
          sliceData.mask_image
        );
      }


      return (
        sliceData.overlay_image
      );
    }


    // -------------------------------------------------------
    // Fallback for older backend response
    // -------------------------------------------------------

    if (
      activeView === "original"
    ) {

      return (
        result.original_image
      );
    }


    if (
      activeView === "mask"
    ) {

      return (
        result.mask_image
      );
    }


    return (
      result.overlay_image ||
      result.image
    );
  };


  // =========================================================
  // Visualization Description
  // =========================================================

  const getViewDescription = () => {

    if (
      activeView === "original"
    ) {

      return (
        "Original FLAIR MRI slice"
      );
    }


    if (
      activeView === "mask"
    ) {

      return (
        "White: Predicted segmentation region"
      );
    }


    return (
      "Red: Predicted segmentation region"
    );
  };


  // =========================================================
  // Download Current Visualization
  // =========================================================

  const downloadImage = () => {

    if (!result) {
      return;
    }


    const image =
      getActiveImage();


    if (!image) {
      return;
    }


    const link =
      document.createElement(
        "a"
      );


    link.href =
      `data:image/png;base64,${image}`;


    if (
      activeView === "original"
    ) {

      link.download =
        `mri_slice_${currentSlice}.png`;

    } else if (
      activeView === "mask"
    ) {

      link.download =
        `predicted_mask_slice_${currentSlice}.png`;

    } else {

      link.download =
        `segmentation_overlay_slice_${currentSlice}.png`;
    }


    link.click();
  };


  // =========================================================
  // Previous Slice
  // =========================================================

  const previousSlice = () => {

    setCurrentSlice(
      (previous) =>
        Math.max(
          0,
          previous - 1
        )
    );
  };


  // =========================================================
  // Next Slice
  // =========================================================

  const nextSlice = () => {

    if (!result) {
      return;
    }


    setCurrentSlice(
      (previous) =>
        Math.min(
          result.total_slices - 1,
          previous + 1
        )
    );
  };


  // =========================================================
  // Jump to Largest Predicted Region
  // =========================================================

  const jumpToLargestRegion = () => {

    if (!result) {
      return;
    }


    setCurrentSlice(
      Number(
        result.largest_tumor_slice
      )
    );
  };


  // =========================================================
  // Generate Written Analysis Report
  // =========================================================

  const generateReport = () => {

    if (!result) {
      return "";
    }


    const volumeAvailable =
      result.predicted_segmentation_volume_cm3 !== null &&
      result.predicted_segmentation_volume_cm3 !== undefined;


    let volumeSentence = "";


    if (volumeAvailable) {

      volumeSentence =
        ` Based on the voxel dimensions stored in the NIfTI file, this corresponds to a predicted segmentation volume of ${Number(
          result.predicted_segmentation_volume_cm3
        ).toLocaleString()} cm³.`;
    }


    return `The uploaded FLAIR MRI volume, named ${result.filename}, was successfully processed using the U-Net segmentation model with a ResNet34 encoder.

The MRI volume contained ${result.total_slices} slices, with dimensions of ${result.volume_shape[0]} × ${result.volume_shape[1]} × ${result.volume_shape[2]} voxels.

The model generated predicted segmentation regions across ${result.predicted_tumor_slices} slices. Among these, slice ${result.largest_tumor_slice} contained the largest predicted segmentation region, covering ${Number(
      result.largest_slice_area_pixels
    ).toLocaleString()} pixels.

Across the complete volume, the model predicted ${Number(
      result.total_predicted_tumor_voxels
    ).toLocaleString()} voxels as part of the segmentation mask.${volumeSentence}

These measurements describe the model's predicted output only. They do not establish a clinical diagnosis, determine tumor type, or indicate malignancy.`;
  };


  // =========================================================
  // Current Slice Statistics
  // =========================================================

  const currentSliceData =
    getCurrentSliceData();


  // =========================================================
  // UI
  // =========================================================

  return (

    <div className="app-layout">


      {/* =====================================================
          LEFT SIDEBAR
      ====================================================== */}

      <aside className="sidebar">


        {/* BRAND */}

        <div className="brand">

          <div className="brand-icon">
            ✚
          </div>

          <div>

            <h2>
              NeuroScan
            </h2>

            <span>
              Brain Imaging Analysis
            </span>

          </div>

        </div>


        {/* MRI ANALYSIS */}

        <div className="sidebar-section">

          <p className="section-label">
            MRI ANALYSIS
          </p>


          <label className="upload-area">

            <div className="upload-icon">
              ↑
            </div>

            <strong>
              Upload FLAIR MRI
            </strong>

            <span>
              Select a NIfTI volume
            </span>

            <span className="file-formats">
              .nii / .nii.gz
            </span>


            <input
              type="file"
              accept=".nii,.nii.gz"
              onChange={handleFileChange}
            />

          </label>


          {/* SELECTED FILE */}

          {file && (

            <div className="selected-file">

              <span>
                Selected file
              </span>

              <strong>
                {file.name}
              </strong>

            </div>

          )}


          {/* RUN BUTTON */}

          <button
            className="predict-button"
            onClick={handlePredict}
            disabled={
              !file ||
              loading
            }
          >

            {
              loading
                ? "Processing MRI..."
                : "Run Segmentation"
            }

          </button>


          {/* LOADING MESSAGE */}

          {loading && (

            <p className="loading-text">
              Processing MRI slices.
              This may take a few minutes.
            </p>

          )}


          {/* ERROR */}

          {error && (

            <p className="error-message">
              {error}
            </p>

          )}

        </div>


        {/* =================================================
            MODEL INFORMATION
        ================================================== */}

        <div className="sidebar-section model-info">

          <p className="section-label">
            MODEL INFORMATION
          </p>


          <div className="info-item">

            <span>
              Architecture
            </span>

            <strong>
              U-Net
            </strong>

          </div>


          <div className="info-item">

            <span>
              Encoder
            </span>

            <strong>
              ResNet34
            </strong>

          </div>


          <div className="info-item">

            <span>
              Input modality
            </span>

            <strong>
              FLAIR
            </strong>

          </div>


          <div className="info-item">

            <span>
              Segmentation
            </span>

            <strong>
              Whole Tumor
            </strong>

          </div>

        </div>


        {/* FOOTER */}

        <div className="sidebar-footer">

          Academic Research Project

          <br />

          Not for clinical diagnosis

        </div>

      </aside>


      {/* =====================================================
          MAIN CONTENT
      ====================================================== */}

      <main className="main-content">


        {/* TOP BAR */}

        <header className="topbar">

          <div>

            <h1>
              Brain Tumor Segmentation
            </h1>

            <p>
              AI-assisted MRI segmentation and analysis
            </p>

          </div>


          <span className="status-badge">
            Research Demo
          </span>

        </header>


        {/* =================================================
            WELCOME
        ================================================== */}

        {!result && !loading && (

          <section className="welcome-card">

            <div className="welcome-icon">
              🧠
            </div>

            <h2>
              Welcome to NeuroScan
            </h2>

            <p>
              Upload a FLAIR MRI volume to generate
              a predicted brain tumor segmentation
              mask and its analysis report.
            </p>

            <span>
              Select your MRI file from the left
              panel to begin.
            </span>

          </section>

        )}


        {/* =================================================
            LOADING
        ================================================== */}

        {loading && (

          <section className="welcome-card">

            <div className="loader"></div>

            <h2>
              Analyzing MRI Volume
            </h2>

            <p>
              The segmentation model is processing
              the MRI slices. Please wait.
            </p>

          </section>

        )}


        {/* =================================================
            RESULTS
        ================================================== */}

        {result && (

          <div className="results-container">


            {/* =============================================
                VISUALIZATION
            ============================================== */}

            <section className="result-card">


              {/* CARD HEADING */}

              <div className="card-heading">

                <div>

                  <h2>
                    Interactive MRI Slice Viewer
                  </h2>

                  <p>
                    Viewing Slice {currentSlice} of{" "}
                    {result.total_slices - 1}
                  </p>

                </div>


                <span className="result-tag">
                  Completed
                </span>

              </div>


              {/* =========================================
                  IMAGE TYPE TABS
              ========================================== */}

              <div className="view-tabs">


                <button
                  className={
                    activeView === "original"
                      ? "view-tab active"
                      : "view-tab"
                  }
                  onClick={() =>
                    setActiveView(
                      "original"
                    )
                  }
                >

                  Original MRI

                </button>


                <button
                  className={
                    activeView === "mask"
                      ? "view-tab active"
                      : "view-tab"
                  }
                  onClick={() =>
                    setActiveView(
                      "mask"
                    )
                  }
                >

                  Predicted Mask

                </button>


                <button
                  className={
                    activeView === "overlay"
                      ? "view-tab active"
                      : "view-tab"
                  }
                  onClick={() =>
                    setActiveView(
                      "overlay"
                    )
                  }
                >

                  Overlay

                </button>

              </div>


              {/* =========================================
                  MRI IMAGE
              ========================================== */}

              <div className="image-container">

                <img
                  src={
                    `data:image/png;base64,${getActiveImage()}`
                  }
                  alt={
                    `${activeView} visualization for MRI slice ${currentSlice}`
                  }
                />

              </div>


              {/* =========================================
                  IMAGE CAPTION
              ========================================== */}

              <div className="image-caption">

                {
                  activeView === "overlay" &&
                  (
                    <span className="red-dot"></span>
                  )
                }


                {
                  activeView === "mask" &&
                  (
                    <span className="white-dot"></span>
                  )
                }


                {getViewDescription()}

              </div>


              {/* =========================================
                  INTERACTIVE SLICE VIEWER
              ========================================== */}

              <div className="slice-viewer-controls">


                {/* SLICE NUMBER */}

                <div className="slice-header">

                  <span>
                    MRI Slice
                  </span>

                  <strong>
                    {currentSlice} /{" "}
                    {result.total_slices - 1}
                  </strong>

                </div>


                {/* SLIDER */}

                

                <div
                  style={{
                    width: "100%",
                    padding: "12px 0",
                  }}
                >
                  <input
                    type="range"
                    min={0}
                    max={result.total_slices - 1}
                    step={1}
                    value={currentSlice}
                    onChange={(event) => {
                      setCurrentSlice(Number(event.target.value));
                    }}
                    style={{
                      display: "block",
                      width: "100%",
                      height: "20px",
                      margin: "0",
                      cursor: "pointer",
                      accentColor: "#287d83",
                      opacity: 1,
                      visibility: "visible",
                    }}
                  />
                </div>


                {/* NAVIGATION BUTTONS */}

                <div className="slice-buttons">


                  <button
                    onClick={
                      previousSlice
                    }
                    disabled={
                      currentSlice === 0
                    }
                  >

                    ← Previous

                  </button>


                  <button
                    className="largest-region-button"
                    onClick={
                      jumpToLargestRegion
                    }
                  >

                    Jump to Largest Region

                  </button>


                  <button
                    onClick={
                      nextSlice
                    }
                    disabled={
                      currentSlice ===
                      result.total_slices - 1
                    }
                  >

                    Next →

                  </button>

                </div>


                {/* =====================================
                    CURRENT SLICE INFORMATION
                ====================================== */}

                {currentSliceData && (

                  <div className="slice-information">

                    <div>

                      <span>
                        Predicted area on this slice
                      </span>

                      <strong>
                        {Number(
                          currentSliceData
                            .predicted_area_pixels
                        ).toLocaleString()}{" "}
                        pixels
                      </strong>

                    </div>


                    <div>

                      <span>
                        Segmentation detected
                      </span>

                      <strong>

                        {
                          currentSliceData.has_prediction
                            ? "Yes"
                            : "No"
                        }

                      </strong>

                    </div>

                  </div>

                )}

              </div>


              {/* DOWNLOAD */}

              <button
                className="download-button"
                onClick={
                  downloadImage
                }
              >

                ↓ Download Current Image

              </button>

            </section>


            {/* =============================================
                PREDICTION REPORT
            ============================================== */}

            <section className="report-card">


              {/* REPORT HEADING */}

              <div className="card-heading">

                <div>

                  <h2>
                    Prediction Report
                  </h2>

                  <p>
                    Summary of the model&apos;s
                    segmentation output
                  </p>

                </div>


                <span className="report-icon">
                  ▤
                </span>

              </div>


              {/* =========================================
                  MAIN STATISTICS
              ========================================== */}

              <div className="statistics-grid">


                <div className="stat-box">

                  <span>
                    Total Slices
                  </span>

                  <strong>
                    {result.total_slices}
                  </strong>

                </div>


                <div className="stat-box">

                  <span>
                    Predicted Slices
                  </span>

                  <strong>
                    {
                      result.predicted_tumor_slices
                    }
                  </strong>

                </div>


                <div className="stat-box">

                  <span>
                    Largest Predicted Region
                  </span>

                  <strong>
                    Slice{" "}
                    {
                      result.largest_tumor_slice
                    }
                  </strong>

                </div>


                <div className="stat-box">

                  <span>
                    Segmentation Volume
                  </span>

                  <strong>

                    {
                      result
                        .predicted_segmentation_volume_cm3 !==
                        null &&
                      result
                        .predicted_segmentation_volume_cm3 !==
                        undefined

                        ? `${Number(
                            result
                              .predicted_segmentation_volume_cm3
                          ).toLocaleString()} cm³`

                        : "Unavailable"
                    }

                  </strong>

                </div>

              </div>


              {/* =========================================
                  WRITTEN ANALYSIS
              ========================================== */}

              <div className="written-report">

                <h3>
                  Analysis Summary
                </h3>

                <p>
                  {generateReport()}
                </p>

              </div>


              {/* =========================================
                  REPORT DETAILS
              ========================================== */}

              <div className="report-details">


                <div>

                  <span>
                    Input file
                  </span>

                  <strong>
                    {result.filename}
                  </strong>

                </div>


                <div>

                  <span>
                    Volume dimensions
                  </span>

                  <strong>

                    {
                      result.volume_shape.join(
                        " × "
                      )
                    }

                  </strong>

                </div>


                <div>

                  <span>
                    Predicted voxels
                  </span>

                  <strong>

                    {
                      Number(
                        result
                          .total_predicted_tumor_voxels
                      ).toLocaleString()
                    }

                  </strong>

                </div>


                <div>

                  <span>
                    Largest slice area
                  </span>

                  <strong>

                    {
                      Number(
                        result
                          .largest_slice_area_pixels
                      ).toLocaleString()
                    }{" "}
                    pixels

                  </strong>

                </div>


                <div>

                  <span>
                    Voxel spacing
                  </span>

                  <strong>

                    {
                      result.voxel_spacing

                        ? `${result.voxel_spacing
                            .map(
                              (value) =>
                                Number(
                                  value
                                ).toFixed(
                                  2
                                )
                            )
                            .join(
                              " × "
                            )} mm`

                        : "Unavailable"
                    }

                  </strong>

                </div>


                <div>

                  <span>
                    Predicted segmentation volume
                  </span>

                  <strong>

                    {
                      result
                        .predicted_segmentation_volume_cm3 !==
                        null &&
                      result
                        .predicted_segmentation_volume_cm3 !==
                        undefined

                        ? `${Number(
                            result
                              .predicted_segmentation_volume_cm3
                          ).toLocaleString()} cm³`

                        : "Unavailable"
                    }

                  </strong>

                </div>

              </div>

            </section>


            {/* =============================================
                RESEARCH DISCLAIMER
            ============================================== */}

            <div className="disclaimer">

              <strong>
                Research Disclaimer
              </strong>

              <p>
                This segmentation is generated by an
                academic deep learning model. The
                predicted regions are not a medical
                diagnosis and should not be used for
                clinical decision-making. Results
                require interpretation by qualified
                medical professionals.
              </p>

            </div>

          </div>

        )}

      </main>

    </div>
  );
}


export default App;