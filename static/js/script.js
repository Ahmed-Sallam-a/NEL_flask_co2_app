document.getElementById('file-input').addEventListener('change', function (event) {
    const file = event.target.files[0];
    const modelType = document.getElementById('model-type').value;
    const originalIndustryImage = document.getElementById('original-industry-image');
    const predictionIndustryImage = document.getElementById('prediction-industry-image');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorMessage = document.getElementById('error-message');

    if (!file) {
        errorMessage.textContent = "Please select a file.";
        errorMessage.style.display = 'block';
        return;
    }

    // Show loading spinner
    loadingSpinner.style.display = 'block';
    errorMessage.style.display = 'none';
    originalIndustryImage.style.display = 'none';
    predictionIndustryImage.style.display = 'none';

    // Prepare form data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', modelType);

    // Send request to Flask API
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => {
        if (!response.ok) {
            return response.text().then(errorData => {
                console.error('Server error:', errorData);
                throw new Error(errorData || 'Unknown error');
            });
        }
        return response.blob();
    })
    .then(blob => {
        if (blob.size === 0) {
            throw new Error('Received empty response');
        }

        if (modelType === "Industry") {
            // Handle ZIP response
            const zipReader = new FileReader();
            zipReader.onload = function (e) {
                const zip = new JSZip();
                zip.loadAsync(e.target.result).then(function (zipContents) {
                    zipContents.file("original.png").async("blob").then(function (originalBlob) {
                        originalIndustryImage.src = URL.createObjectURL(originalBlob);
                        originalIndustryImage.style.display = 'block';
                    });

                    zipContents.file("prediction.png").async("blob").then(function (predictionBlob) {
                        predictionIndustryImage.src = URL.createObjectURL(predictionBlob);
                        predictionIndustryImage.style.display = 'block';
                    });
                });
            };
            zipReader.readAsArrayBuffer(blob);
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        errorMessage.textContent = 'An error occurred. Please try again.';
        errorMessage.style.display = 'block';
    })
    .finally(() => {
        loadingSpinner.style.display = 'none';
    });
});
