// --------- Sidebar Active Link ---------
document.addEventListener("DOMContentLoaded", () => {
    const links = document.querySelectorAll(".sidebar a");
    const currentPath = window.location.pathname;

    links.forEach(link => {
        if (link.getAttribute("href") === currentPath) {
            link.style.backgroundColor = "#343454";
            link.style.color = "#fff";
        }
    });
});

// --------- Prediction Function ---------
function predictJob(event) {
    event.preventDefault();

    const jobTitle = document.getElementById("job_title").value.trim();
    const jobDescription = document.getElementById("job_description").value.trim();
    const requirements = document.getElementById("requirements").value.trim();
    const salary = document.getElementById("salary").value.trim();

    // Basic frontend validation
    if (jobTitle.length < 3 || jobDescription.length < 10) {
        alert("Please enter sufficient job details.");
        return;
    }

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            job_title: jobTitle,
            job_description: jobDescription,
            requirements: requirements,
            salary: salary
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        // ----- Elements -----
        const resultPanel = document.getElementById("resultPanel");
        const resultText = document.getElementById("resultText");
        const confidenceText = document.getElementById("confidenceText");
        const progressCircle = document.querySelector(".meter .progress");

        // Show panel
        resultPanel.style.display = "block";

        // Prediction text
        resultText.innerText = "Prediction: " + data.prediction;

        // Color logic
        if (data.prediction === "Fake Job") {
            resultText.style.color = "#e53935";
            progressCircle.style.stroke = "#e53935";
        } else {
            resultText.style.color = "#43a047";
            progressCircle.style.stroke = "#43a047";
        }

        // Confidence animation
        const confidence = parseFloat(data.confidence);
        confidenceText.innerText = confidence + "%";

        const maxStroke = 440;   // circumference
        const offset = maxStroke - (confidence / 100) * maxStroke;

        // Reset then animate
        progressCircle.style.strokeDashoffset = maxStroke;
        setTimeout(() => {
            progressCircle.style.strokeDashoffset = offset;
        }, 100);

        // Scroll into view
        resultPanel.scrollIntoView({ behavior: "smooth" });
    })
    .catch(error => {
        console.error("Prediction error:", error);
        alert("Server error. Please try again.");
    });
}
