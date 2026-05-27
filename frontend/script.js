document.getElementById("flagForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById("file");
    const resultElement = document.getElementById('prediction');

    if (fileInput.files.length === 0){
        alert("Please select a flag image first.")
        return;
    }

    resultElement.innerText = "Analyzing flag layout...";

    const formData = new FormData(this);

    try {
        const response = await fetch(`https://country-flag-classifier-api.onrender.com/predict-image`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok){
            throw new Error(`Server returned code ${response.status}`);
        }

        const data = await response.json()

        resultElement.innerText = `Predicted Country: ${data.country} (${((data.confidence) * 100).toFixed(1)}%)`;
    } catch (error){
        console.error("Error identifying flag", error);
        resultElement.innerText = "Failed to recognize flag. Please try another image.";
    }

});