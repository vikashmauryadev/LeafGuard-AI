
import { useEffect, useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [gradcamUrl, setGradcamUrl] = useState("");
  const [explanationError, setExplanationError] = useState("");
  const [history, setHistory] = useState([]);
  const [showAllHistory, setShowAllHistory] = useState(false);
  const [historySearch, setHistorySearch] = useState("");
  const [historyStatus, setHistoryStatus] = useState("all");
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState("");
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState("");

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  useEffect(() => {
    fetchHistory();
  }, []);
  useEffect(() => {
    fetchStats();
  }, []);

  useEffect(() => {
    return () => {
      if (gradcamUrl) URL.revokeObjectURL(gradcamUrl);
    };
  }, [gradcamUrl]);
  async function fetchHistory() {
    setHistoryLoading(true);
    setHistoryError("");

    try {
      const response = await fetch(`${API_URL}/history?limit=100`);

      if (!response.ok) {
        throw new Error("Unable to load prediction history.");
      }

      const data = await response.json();
      setHistory(data.history || []);
    } catch (err) {
      setHistoryError(
        err.message || "Could not connect to the history API."
      );
    } finally {
      setHistoryLoading(false);
    }
  }
  async function fetchStats() {
    setStatsLoading(true);
    setStatsError("");

    try {
      const response = await fetch(`${API_URL}/stats`);

      if (!response.ok) {
        throw new Error("Unable to load dashboard statistics.");
      }

      const data = await response.json();
      setStats(data);
    } catch (err) {
      setStatsError(
        err.message || "Could not connect to the statistics API."
      );
    } finally {
      setStatsLoading(false);
    }
  }

  function handleImageChange(event) {
    const file = event.target.files?.[0];
    event.target.value = "";

    if (!file) return;

    setError("");
    setResult(null);
    setGradcamUrl("");
    setExplanationError("");

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("Please choose an image smaller than 10 MB.");
      return;
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
  }

  function handleReset() {
    setImage(null);
    setPreview("");
    setResult(null);
    setError("");
    setGradcamUrl("");
    setExplanationError("");
    setLoading(false);
  }

  async function handleAnalyze() {
    if (!image || loading) {
      if (!image) setError("Upload a leaf image before analysis.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setGradcamUrl("");
    setExplanationError("");

    try {
      const formData = new FormData();
      formData.append("file", image);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Image analysis failed. Please try another image."
        );
      }

      setResult(data);

      // Generate the visual explanation separately so it cannot break prediction.
      try {
        const explainForm = new FormData();
        explainForm.append("file", image);

        const explainResponse = await fetch(`${API_URL}/explain`, {
          method: "POST",
          body: explainForm,
        });

        if (!explainResponse.ok) {
          const details = await explainResponse.text();
          throw new Error(
            details || "The visual explanation could not be generated."
          );
        }

        const explanationBlob = await explainResponse.blob();
        if (!explanationBlob.type.startsWith("image/")) {
          throw new Error("The explanation endpoint did not return an image.");
        }

        setGradcamUrl(URL.createObjectURL(explanationBlob));
      } catch (explainErr) {
        setExplanationError(
          explainErr.message ||
          "The visual explanation could not be generated. Please try again."
        );
      }
    } catch (err) {
      setError(
        err.message ||
        "Unable to connect to the backend. Make sure the API is running."
      );
    } finally {
      setLoading(false);
    }
  }
    
  async function handleDownloadReport() {
    if (!image || reportLoading) {
      return;
    }

    setReportLoading(true);
    setReportError("");

    try {
      const formData = new FormData();
      formData.append("file", image);

      const response = await fetch(`${API_URL}/report`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let message = "Unable to generate the PDF report.";

        try {
          const data = await response.json();
          if (typeof data.detail === "string") {
            message = data.detail;
          }
        } catch {
          // Keep the default error message.
        }

        throw new Error(message);
      }

      const pdfBlob = await response.blob();

      if (
        pdfBlob.type !== "application/pdf" &&
        pdfBlob.size > 0
      ) {
        throw new Error("The server did not return a valid PDF.");
      }

      if (pdfBlob.size === 0) {
        throw new Error("The generated PDF is empty.");
      }

      const downloadUrl = URL.createObjectURL(pdfBlob);
      const link = document.createElement("a");

      link.href = downloadUrl;
      link.download = "leafguard_prediction_report.pdf";
      document.body.appendChild(link);
      link.click();
      link.remove();

      setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
    } catch (err) {
      setReportError(
        err.message || "Could not download the report. Please try again."
      );
    } finally {
      setReportLoading(false);
    }
  
    
  
  
  }

  const confidence = Math.min(
    100,
    Math.max(0, Number(result?.confidence_percent ?? 0))
  );

  const isUncertain = result?.status === "uncertain";
  const filteredHistory = history.filter((item) => {
    const searchText = historySearch.trim().toLowerCase();

    const diseaseName = (
      item.display_name || item.predicted_class || ""
    ).toLowerCase();

    const fileName = (item.filename || "").toLowerCase();

    const matchesSearch =
      diseaseName.includes(searchText) || fileName.includes(searchText);

    const matchesStatus =
      historyStatus === "all" ||
      (historyStatus === "uncertain"
        ? item.status === "uncertain"
        : item.status !== "uncertain");

    return matchesSearch && matchesStatus;
  });
const visibleHistory = showAllHistory
  ? filteredHistory
  : filteredHistory.slice(0, 10);
  return (
    <div className="app-shell">
      <style>{`
        .lg-result-tag {
          display: inline-block;
          padding: 7px 11px;
          border-radius: 999px;
          background: #e4f5e9;
          color: #176638;
          font-size: 11px;
          font-weight: 800;
          letter-spacing: .06em;
        }
        .lg-result-tag.warning {
          background: #fff1d5;
          color: #8a5700;
        }
        .lg-confidence { margin: 22px 0; }
        .lg-confidence-heading {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          margin-bottom: 10px;
          font-size: 13px;
        }
        .lg-confidence-heading strong { font-size: 20px; }
        .lg-track {
          height: 9px;
          overflow: hidden;
          border-radius: 20px;
          background: #e7eee8;
        }
        .lg-fill {
          height: 100%;
          border-radius: inherit;
          background: #27834c;
          transition: width .35s ease;
        }
        .lg-fill.warning { background: #d18a16; }
        .lg-summary { line-height: 1.7; }
        .lg-advisory { margin-top: 22px; }
        .lg-advisory h4 { margin-bottom: 12px; }
        .lg-advisory ul {
          padding-left: 21px;
          line-height: 1.8;
        }
        .lg-advisory li { margin-bottom: 8px; }
        .lg-warning {
          margin-top: 18px;
          padding: 15px;
          border: 1px solid #efd49a;
          border-radius: 12px;
          background: #fff9e9;
          color: #76500d;
          line-height: 1.65;
        }
        .lg-warning p { margin: 8px 0 0; }
        .lg-disclaimer {
          margin-top: 20px;
          padding-top: 14px;
          border-top: 1px solid #e6ebe6;
          color: #6b756e;
          font-size: 12px;
          line-height: 1.7;
        }
        .lg-error {
          margin-top: 12px;
          padding: 12px;
          border-radius: 10px;
          background: #fff0ee;
          color: #a52b20;
          font-size: 13px;
          line-height: 1.5;
        }
        .lg-analyze:disabled {
          opacity: .6;
          cursor: not-allowed;
        }
        .lg-preview {
          display: block;
          width: 100%;
          max-height: 320px;
          object-fit: contain;
          border-radius: 12px;
        }
        .lg-gradcam {
          margin-top: 22px;
          padding-top: 18px;
          border-top: 1px solid #e6ebe6;
        }
        .lg-gradcam h4 { margin: 0 0 8px; }
        .lg-gradcam p {
          font-size: 13px;
          line-height: 1.6;
          color: #6b756e;
        }
        .lg-gradcam-image {
          display: block;
          width: 100%;
          max-height: 340px;
          object-fit: contain;
          border-radius: 12px;
          border: 1px solid #e6ebe6;
          background: #f7faf7;
        }
        .lg-reset {
          margin-top: 12px;
          border: 0;
          background: transparent;
          color: #27834c;
          cursor: pointer;
          font-weight: 600;
        }
        .lg-placeholder {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 10px;
          padding: 35px 12px;
          text-align: center;
        }
        .lg-upload-icon {
          display: grid;
          place-items: center;
          width: 48px;
          height: 48px;
          border-radius: 14px;
          background: #e4f5e9;
          color: #176638;
          font-size: 28px;
        }
        .lg-file-name {
          display: block;
          margin-top: 12px;
          overflow-wrap: anywhere;
          font-size: 13px;
        }
        .lg-empty-icon {
          margin-bottom: 14px;
          color: #27834c;
          font-size: 42px;
          text-align: center;
        }
        .lg-placeholder-row {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          margin-top: 15px;
          padding-top: 12px;
          border-top: 1px solid #e6ebe6;
          font-size: 13px;
        }
        .lg-status {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 9px 12px;
          border-radius: 999px;
          background: #e4f5e9;
          color: #176638;
          font-size: 12px;
          font-weight: 600;
        }
        .lg-status-dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: #27834c;
        }
        @media (max-width: 600px) {
          .lg-confidence-heading { align-items: flex-start; }
        }
      `}</style>

      <aside className="sidebar">
        <a className="brand" href="#home">
          <span className="brand-icon">✳</span>
          <span>LeafGuard<span className="brand-ai"> AI</span></span>
        </a>

        <p className="nav-heading">WORKSPACE</p>
        <a className="nav-link active" href="#home">▦ <span>Dashboard</span></a>
        <a className="nav-link" href="#analyzer">⌕ <span>Disease Analyzer</span></a>
        <a className="nav-link" href="#prediction-history">
          ◷ <span>Prediction History</span>
        </a>
        <a className="nav-link" href="#how-it-works">⌘ <span>How it works</span></a>

        <div className="sidebar-bottom">
          <div className="status-dot" />
          <div>
            <strong>Plant health AI</strong>
            <p>MobileNetV2 classification</p>
          </div>
        </div>
      </aside>

      <main className="main-content" id="home">
        <header className="topbar">
          <div>
            <p className="eyebrow">PLANT HEALTH INTELLIGENCE</p>
            <h1>Plant Disease Dashboard</h1>
          </div>
          <span className="lg-status">
            <span className="lg-status-dot" />
            AI workspace
          </span>
        </header>

        <section className="welcome-banner">
          <div className="welcome-copy">
            <span className="banner-label">
              LEAFGUARD AI · SMART AGRICULTURE
            </span>
            <h2>Healthier leaves.<br />Smarter decisions.</h2>
            <p>
              Analyze a plant leaf image using a trained machine learning
              model and review the associated plant-care guidance.
            </p>
            
<a className="primary-link" href="#analyzer">
  Analyze a Leaf <span aria-hidden="true">→</span>
</a>
          </div>
          <div className="leaf-art" aria-hidden="true">
            <div className="leaf-circle circle-one" />
            <div className="leaf-circle circle-two" />
            <div className="leaf-symbol">❋</div>
            <span className="art-caption">GROW WITH INSIGHT</span>
          </div>
        </section>


        <section className="stats-grid" aria-label="Live project statistics">
          <article className="stat-card">
            <span className="stat-icon">⌁</span>
            <p>Total predictions</p>
            <h3>
              {statsLoading ? "..." : stats ? stats.total_predictions : "—"}
            </h3>
            <span className="stat-note">
              Images analyzed and saved
            </span>
          </article>

          <article className="stat-card">
            <span className="stat-icon">◉</span>
            <p>Disease classes</p>
            <h3>
              {statsLoading ? "..." : stats ? stats.unique_classes : "—"}
            </h3>
            <span className="stat-note">
              Unique classes in prediction history
            </span>
          </article>

          <article className="stat-card">
            <span className="stat-icon">✧</span>
            <p>Average confidence</p>
            <h3>
              {statsLoading
                ? "..."
                : stats && stats.average_confidence_percent !== null
                  ? `${stats.average_confidence_percent}%`
                  : "—"}
            </h3>
            <span className="stat-note">
              Mean model confidence across saved predictions
            </span>
          </article>

          <article className="stat-card">
            <span className="stat-icon">!</span>
            <p>Uncertain predictions</p>
            <h3>
              {statsLoading ? "..." : stats ? stats.uncertain_predictions : "—"}
            </h3>
            <span className="stat-note">
              Predictions flagged as uncertain
            </span>
          </article>

          {statsError && (
            <p role="alert" className="stat-note">
              {statsError}
              {" "}
              <button type="button" onClick={fetchStats}>
                Retry
              </button>
            </p>
          )}
        </section>

        <section className="analyzer-section" id="analyzer">
          <div className="section-heading">
            <div>
              <p className="eyebrow">IMAGE WORKSPACE</p>
              <h2>Leaf disease analyzer</h2>
              <p className="section-description">
                Choose a clear, well-lit photograph of a plant leaf.
              </p>
            </div>
            <span className="step-badge">01 / ANALYZE</span>
          </div>

          <div className="analyzer-grid" id="analyzer">
  <article className="panel upload-panel">
    <h3>Upload leaf image</h3>
              <p className="muted">
                Keep the leaf in focus and avoid blurry photographs.
              </p>

              <label className="upload-area">
                {preview ? (
                  <img
                    className="lg-preview"
                    src={preview}
                    alt="Selected plant leaf"
                  />
                ) : (
                  <span className="lg-placeholder">
                    <span className="lg-upload-icon">↑</span>
                    <strong>Choose an image to upload</strong>
                    <span>Click here to browse your files</span>
                    <small>JPG, PNG or WebP · Maximum 10 MB</small>
                  </span>
                )}
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleImageChange}
                />
              </label>

              {image && (
                <div>
                  <span className="lg-file-name">{image.name}</span>
                  <button
                    type="button"
                    className="lg-reset"
                    onClick={handleReset}
                    disabled={loading}
                  >
                    Remove image
                  </button>
                </div>
              )}

              {error && (
                <p className="lg-error" role="alert">{error}</p>
              )}

              <button
                type="button"
                className="analyze-button lg-analyze"
                onClick={handleAnalyze}
                disabled={!image || loading}
              >
                {loading ? "Analyzing image..." : "Analyze leaf"}
                {!loading && <span> →</span>}
              </button>

              <p className="privacy-note">
                The selected image is sent to your configured backend for
                analysis.
              </p>
            </article>

            <article className="panel result-panel" aria-live="polite">
              <div className="result-title">
                <div>
                  <p className="eyebrow">ANALYSIS OUTPUT</p>
                  <h3>Detection results</h3>
                </div>
                <span className="result-icon">✧</span>
              </div>

              {loading ? (
                <div className="empty-result">
                  <div className="lg-empty-icon">◌</div>
                  <h4>Analyzing your image</h4>
                  <p>
                    The trained model is processing the leaf photograph.
                    Please wait.
                  </p>
                </div>
              ) : result ? (
                <div className="result-content">
                  <span className={`lg-result-tag ${isUncertain ? "warning" : ""}`}>
                    {isUncertain
                      ? "PREDICTION NEEDS REVIEW"
                      : "AI PREDICTION"}
                  </span>

                  <h2 style={{ marginTop: 16 }}>
                    {result.display_name || result.predicted_class}
                  </h2>

                  <div className="lg-confidence">
                    <div className="lg-confidence-heading">
                      <span>Model confidence output</span>
                      <strong>{confidence.toFixed(2)}%</strong>
                    </div>
                    <div
                      className="lg-track"
                      role="progressbar"
                      aria-label="Model confidence output"
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-valuenow={confidence}
                    >
                      <div
                        className={`lg-fill ${isUncertain ? "warning" : ""}`}
                        style={{ width: `${confidence}%` }}
                      />
                    </div>
                  </div>

                  <p className="lg-summary">
                    {result.advisory?.summary ||
                      "Review the predicted class alongside visible symptoms."}
                  </p>

                  {isUncertain && (
                    <div className="lg-warning" role="status">
                      <strong>Low-confidence result</strong>
                      <p>
                        {result.message ||
                          "Try a clearer photograph or seek expert advice."}
                      </p>
                    </div>
                  )}

                  <div className="lg-advisory">
                    <h4>Recommended next steps</h4>
                    <ul>
                      {(result.advisory?.actions || []).map((action, index) => (
                        <li key={`${index}-${action}`}>{action}</li>
                      ))}
                    </ul>
                  </div>

                  <p className="lg-disclaimer">
                    {result.disclaimer ||
                      "AI screening is not a definitive plant disease diagnosis."}
                  </p>
<div style={{ marginTop: 18 }}>
  <button
    type="button"
    className="analyze-button"
    onClick={handleDownloadReport}
    disabled={reportLoading || !image}
  >
    {reportLoading ? "Generating PDF..." : "Download PDF Report"}
  </button>

  {reportError && (
    <p className="lg-error" role="alert">
      {reportError}
    </p>
  )}
</div>
                  <section className="lg-gradcam" aria-label="Visual explanation">
                    <h4>Visual explanation (Grad-CAM)</h4>
                    <p>
                      The overlay highlights image regions that influenced the
                      model's prediction. It is an aid to interpretation, not
                      proof that a specific region contains disease.
                    </p>
                    {gradcamUrl ? (
                      <img
                        className="lg-gradcam-image"
                        src={gradcamUrl}
                        alt="Grad-CAM heatmap overlay for the analyzed leaf"
                      />
                    ) : explanationError ? (
                      <p className="lg-error" role="alert">
                        Grad-CAM unavailable: {explanationError}
                      </p>
                    ) : (
                      <p role="status">Generating visual explanation...</p>
                    )}
                  </section>
                </div>
              ) : (
                <div className="empty-result">
                  <div className="scan-illustration">
                    <span className="scan-leaf">❋</span>
                    <span className="scan-line" />
                  </div>
                  <h4>Your results will appear here</h4>
                  <p>
                    Upload a leaf photograph to see the model prediction,
                    confidence output and plant-care guidance.
                  </p>
                  <div className="lg-placeholder-row">
                    <span>Predicted class</span><span>—</span>
                  </div>
                  <div className="lg-placeholder-row">
                    <span>Confidence</span><span>—</span>
                  </div>
                </div>
              )}
            </article>
          </div>
        </section>

        <section
          className="how-section"
          id="prediction-history"
          style={{ marginBottom: 40 }}
        >
          <div className="section-heading">
            <div>
              <p className="eyebrow">RECENT ACTIVITY</p>
              <h2>Prediction History</h2>
              <p className="section-description">
                Review your recently analyzed plant leaf images.
              </p>
              <input
  type="text"
  placeholder="Search disease or filename..."
  value={historySearch}
  onChange={(event) => setHistorySearch(event.target.value)}
  style={{
    width: "100%",
    maxWidth: 350,
    padding: "12px",
    marginTop: 12,
    border: "1px solid #d5ddd6",
    borderRadius: 8,
    fontSize: 14,
  }}
/>
              <select
                aria-label="Filter prediction history by status"
                value={historyStatus}
                onChange={(event) => setHistoryStatus(event.target.value)}
                style={{
                  display: "block",
                  width: "100%",
                  maxWidth: 350,
                  padding: "12px",
                  marginTop: 10,
                  border: "1px solid #d5ddd6",
                  borderRadius: 8,
                  fontSize: 14,
                  background: "#fff",
                }}
              >
                <option value="all">All predictions</option>
                <option value="predicted">AI predictions</option>
                <option value="uncertain">Needs review</option>
              </select>
            </div>

            <button
              type="button"
              className="analyze-button"
              onClick={fetchHistory}
              disabled={historyLoading}
            >
              {historyLoading ? "Loading..." : "Refresh history"}
            </button>
            <button
  type="button"
  onClick={() => {
    window.location.href = `${API_URL}/history/export`;
  }}
>
  Export CSV
</button>
          </div>

          {historyError && (
            <p className="lg-error" role="alert">
              {historyError}
            </p>
          )}

          {historyLoading && history.length === 0 ? (
            <p className="muted">Loading prediction history...</p>
          ) : history.length === 0 ? (
            <div className="panel">
              <p>No saved predictions yet. Analyze a leaf image to get started.</p>
            </div>
          ) : (
            <div className="panel" style={{ overflowX: "auto" }}>
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  textAlign: "left",
                  minWidth: 600,
                }}
              >
                <thead>
                  <tr>
                    <th style={{ padding: 12 }}>Disease</th>
                    <th style={{ padding: 12 }}>Confidence</th>
                    <th style={{ padding: 12 }}>Status</th>
                    <th style={{ padding: 12 }}>Date</th>
                  </tr>
                </thead>

                <tbody>
                  {visibleHistory.map((item) => (
                    <tr key={item.id}>
                      <td style={{ padding: 12 }}>
                        <strong>{item.display_name || item.predicted_class}</strong>
                        <br />
                        <small>{item.filename || "Image"}</small>
                      </td>

                      <td style={{ padding: 12 }}>
                        {Number(item.confidence_percent).toFixed(2)}%
                      </td>

                      <td style={{ padding: 12 }}>
                        {item.status === "uncertain"
                          ? "Needs review"
                          : "AI prediction"}
                      </td>

                      <td style={{ padding: 12 }}>
                        {item.timestamp
                          ? new Date(item.timestamp).toLocaleString()
                          : "—"}
                      </td>
                    </tr>
                  ))} 
                </tbody>
              </table>                {filteredHistory.length > 10 && (
                <div style={{ textAlign: "center", marginTop: 16 }}>
                  <button
                    type="button"
                    className="analyze-button"
                    onClick={() =>
                      setShowAllHistory((current) => !current)
                    }
                  >
                    {showAllHistory ? "Show Less" : "Show All History"}
                  </button>
                </div>
              )}
            </div> 
               
            
          )}
        </section>

        <section className="how-section" id="how-it-works">
          <div>
            <p className="eyebrow">THE PROCESS</p>
            <h2>From image to insight</h2>
          </div>
          <div className="process-grid">
            <article>
              <span>01</span>
              <h3>Upload</h3>
              <p>Select a clear leaf photograph.</p>
            </article>
            <article>
              <span>02</span>
              <h3>Classify</h3>
              <p>MobileNetV2 processes the image.</p>
            </article>
            <article>
              <span>03</span>
              <h3>Review</h3>
              <p>Read the prediction and relevant care guidance.</p>
            </article>
          </div>
        </section>

        <footer>
          <span>LeafGuard AI</span>
          <span>ML-assisted screening · Not a definitive diagnosis</span>
        </footer>
      </main>
    </div>
  );
}

export default App;