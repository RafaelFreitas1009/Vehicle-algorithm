document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("form");
    const btn = form.querySelector("button");
  
    form.addEventListener("submit", async (e) => {
      e.preventDefault(); // 🚫 impede reload
  
      btn.disabled = true;
      btn.innerText = "Executando...";
  
      const formData = new FormData(form);
  
      try {
        const response = await fetch("/api/run", {
          method: "POST",
          body: formData
        });
  
        if (!response.ok) {
          throw new Error("Erro ao executar análise");
        }
  
        const data = await response.json();
  
        updateMetrics(data);
        updatePlots(data.ts);
  
      } catch (err) {
        alert(err.message);
      } finally {
        btn.disabled = false;
        btn.innerText = "Executar análise";
      }
    });
  });
  
  function updateMetrics(data) {
    document.getElementById("samples").innerText = data.meta.n_samples;
    document.getElementById("rf-acc").innerText = data.rf_accuracy.toFixed(4);
    document.getElementById("et-acc").innerText = data.et_accuracy.toFixed(4);
  }
  
  function updatePlots(ts) {
    const plots = [
      "decision_boundary.png",
      "confusion_rf.png",
      "feature_importance.png",
      "probability_distribution.png",
      "distribution_3d.png",
      "model_comparison.png"
    ];
  
    plots.forEach(name => {
      const img = document.querySelector(`img[data-plot="${name}"]`);
      if (img) {
        img.src = `/static/plots/${name}?t=${ts}`;
      }
    });
  }
  