document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("uploadForm").addEventListener("submit", async function (event) {
        event.preventDefault();

        let fileInput = document.getElementById("file");
        let resultDiv = document.getElementById("result");
        let resultsTable = document.querySelector("#resultsTable tbody");

        if (fileInput.files.length === 0) {
            alert("Please select a file!");
            return;
        }

        let formData = new FormData();
        formData.append("file", fileInput.files[0]);

        try {
            let response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            let data = await response.json();

            if (data.error) {
                resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                return;
            }

            // Clear previous results
            resultsTable.innerHTML = "";

            // Populate table with results
            data.forEach(row => {
                let tr = document.createElement("tr");
                tr.innerHTML = `<td>${row.id}</td><td>${row.is_fraud}</td><td>${row.probability.toFixed(4)}</td>`;
                resultsTable.appendChild(tr);
            });

        } catch (error) {
            resultDiv.innerHTML = `<p style="color: red;">Error processing the file</p>`;
        }
    });
});
